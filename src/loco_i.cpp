#include "loco_i.hpp"
#include "../include/predictors/med_predictor.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// --- LocoIEncoder Implementation ---

LocoIEncoder::LocoIEncoder() : predictor_(std::make_unique<MEDPredictor>()) {
    resetContexts();
}

void LocoIEncoder::resetContexts() {
    N_.assign(NUM_CONTEXTS, 1);
    A_.assign(NUM_CONTEXTS, std::max(2, (int)std::floor((RANGE + 32)/64.0)));
    B_.assign(NUM_CONTEXTS, 0);
    C_.assign(NUM_CONTEXTS, 0);
}

int LocoIEncoder::quantizeGradient(const int gradient) {
    if (gradient <= -T3) return -4;
    if (gradient <= -T2) return -3;
    if (gradient <= -T1) return -2;
    if (gradient < 0)    return -1;
    if (gradient == 0)   return 0;
    if (gradient < T1)   return 1;
    if (gradient < T2)   return 2;
    if (gradient < T3)   return 3;
    return 4;
}

void LocoIEncoder::getContext(const int d1, const int d2, const int d3, int& Q, int& sign) {
    int q1 = quantizeGradient(d1);
    int q2 = quantizeGradient(d2);
    int q3 = quantizeGradient(d3);

    if (q1 < 0 || (q1 == 0 && q2 < 0) || (q1 == 0 && q2 == 0 && q3 < 0)) {
        q1 = -q1;
        q2 = -q2;
        q3 = -q3;
        sign = -1;
    } else {
        sign = 1;
    }

    Q = q1 * 81 + q2 * 9 + q3;
}

int LocoIEncoder::computeGolombK(const int Q) const {
    int k = 0;
    while ((N_[Q] << k) < A_[Q]) {
        k++;
    }
    return k;
}

void LocoIEncoder::encodeMappedError(BitWriter& writer, const int mapped_error, const int k) {
    const int q = mapped_error >> k;

    if (q < LIMIT - qbpp - 1) {
        writer.writeBits((1 << q) - 1, q);
        writer.writeBits(0, 1);
        writer.writeBits(mapped_error, k);
    } else {
        writer.writeBits((1 << (LIMIT - qbpp - 1)) - 1, LIMIT - qbpp - 1);
        writer.writeBits(0, 1);
        writer.writeBits(mapped_error - 1, qbpp);
    }
}

int LocoIEncoder::decodeMappedError(BitReader& reader, const int k) {
    int q = 0;
    // Count leading 1s for unary part
    while (reader.readBits(1) == 1) {
        q++;
    }

    int mapped_error = 0;
    if (q < LIMIT - qbpp - 1) {
        const int remainder = static_cast<int>(reader.readBits(k));
        mapped_error = (q << k) | remainder;
    } else {
        const int val = static_cast<int>(reader.readBits(qbpp));
        mapped_error = val + 1;
    }
    return mapped_error;
}

void LocoIEncoder::updateContext(const int Q, const int error) {
    B_[Q] += error * (2 * RANGE - 1);
    A_[Q] += std::abs(error);
    if (N_[Q] == RESET) {
        A_[Q] = A_[Q] >> 1;
        B_[Q] = B_[Q] >> 1;
        N_[Q] = N_[Q] >> 1;
    }
    N_[Q]++;

    if (B_[Q] <= -N_[Q]) {
        B_[Q] += N_[Q];
        if (C_[Q] > -128) C_[Q]--;
        if (B_[Q] <= -N_[Q]) B_[Q] = -N_[Q] + 1;
    } else if (B_[Q] > 0) {
        B_[Q] -= N_[Q];
        if (C_[Q] < 127) C_[Q]++;
        if (B_[Q] > 0) B_[Q] = 0;
    }
}

std::vector<uint8_t> LocoIEncoder::encode(const std::vector<uint8_t>& image_data, const int width, const int height) {
    if (image_data.size() < width * height) {
        throw std::invalid_argument("Image data size does not match width * height");
    }

    resetContexts();
    BitWriter writer;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int x_val = image_data[y * width + x];

            const int a = (x > 0) ? image_data[y * width + x - 1] : ((y > 0) ? image_data[(y - 1) * width + x] : 0);
            const int b = (y > 0) ? image_data[(y - 1) * width + x] : 0;
            const int c = (x > 0 && y > 0) ? image_data[(y - 1) * width + x - 1] : 0;
            const int d = (x < width - 1 && y > 0) ? image_data[(y - 1) * width + x + 1] : b;

            const int d1 = d - b;
            const int d2 = b - c;
            const int d3 = c - a;

            int Q, sign;
            getContext(d1, d2, d3, Q, sign);

            int px = predictor_->predict(a, b, c, image_data.data(), x, y, width);

            px += sign * C_[Q];

            if (px > MAXVAL) px = MAXVAL;
            if (px < 0) px = 0;

            int error = x_val - px;

            if (error < -MAXVAL / 2) error += RANGE;
            else if (error > (MAXVAL + 1) / 2 - 1) error -= RANGE;

            error = sign * error;

            const int k = computeGolombK(Q);

            int mapped_error;
            const int err_val = (k == 0 && (2 * B_[Q] <= -N_[Q])) ? -error - 1 : error;

            if (err_val >= 0) mapped_error = 2 * err_val;
            else mapped_error = -2 * err_val - 1;

            encodeMappedError(writer, mapped_error, k);
            updateContext(Q, error);
        }
    }

    writer.flush();
    return writer.getBuffer();
}

std::vector<uint8_t> LocoIEncoder::decode(const std::vector<uint8_t>& encoded_data, const int width, const int height) {
    resetContexts();
    BitReader reader(encoded_data);
    std::vector<uint8_t> decoded_image(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            // Reconstruct neighbors from already decoded image
            const int a = (x > 0) ? decoded_image[y * width + x - 1] : ((y > 0) ? decoded_image[(y - 1) * width + x] : 0);
            const int b = (y > 0) ? decoded_image[(y - 1) * width + x] : 0;
            const int c = (x > 0 && y > 0) ? decoded_image[(y - 1) * width + x - 1] : 0;
            const int d = (x < width - 1 && y > 0) ? decoded_image[(y - 1) * width + x + 1] : b;

            const int d1 = d - b;
            const int d2 = b - c;
            const int d3 = c - a;

            int Q, sign;
            getContext(d1, d2, d3, Q, sign);

            int px = predictor_->predict(a, b, c, decoded_image.data(), x, y, width);

            px += sign * C_[Q];

            if (px > MAXVAL) px = MAXVAL;
            if (px < 0) px = 0;

            const int k = computeGolombK(Q);

            // Decode mapped error
            const int mapped_error = decodeMappedError(reader, k);

            // Unmap error
            int err_val;
            if (mapped_error % 2 == 0) {
                err_val = mapped_error / 2;
            } else {
                err_val = -(mapped_error + 1) / 2;
            }

            int error = (k == 0 && (2 * B_[Q] <= -N_[Q])) ? -err_val - 1 : err_val;

            // Update context using the exact same error that was encoded
            updateContext(Q, error);

            // Reconstruct pixel
            error = sign * error;

            int rx = px + error;

            // Modulo reduction unmap
            if (rx < 0) rx += RANGE;
            else if (rx > MAXVAL) rx -= RANGE;

            decoded_image[y * width + x] = static_cast<uint8_t>(rx);
        }
    }

    return decoded_image;
}
