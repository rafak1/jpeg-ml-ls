#include "bit_stream.hpp"

#include <cstdint>

// --- BitWriter Implementation ---

void BitWriter::writeBits(const uint32_t bits, const int count) {
    if (count == 0) return;
    bit_buffer_ = (bit_buffer_ << count) | (bits & ((1U << count) - 1));
    bit_count_ += count;

    while (bit_count_ >= 8) {
        bit_count_ -= 8;
        uint8_t byte = (bit_buffer_ >> bit_count_) & 0xFF;
        buffer_.push_back(byte);

        // JPEG bit stuffing
        if (byte == 0xFF) {
            buffer_.push_back(0x00);
        }
    }
}

void BitWriter::flush() {
    if (bit_count_ > 0) {
        const uint8_t byte = (bit_buffer_ << (8 - bit_count_)) & 0xFF;
        buffer_.push_back(byte);
        if (byte == 0xFF) {
            buffer_.push_back(0x00);
        }
        bit_count_ = 0;
    }
}

const std::vector<uint8_t>& BitWriter::getBuffer() const {
    return buffer_;
}

// --- BitReader Implementation ---

BitReader::BitReader(const std::vector<uint8_t>& buffer) : buffer_(buffer) {
    fillBuffer();
}

void BitReader::fillBuffer() {
    while (bit_count_ <= 24 && byte_pos_ < buffer_.size()) {
        const uint8_t byte = buffer_[byte_pos_++];
        bit_buffer_ = (bit_buffer_ << 8) | byte;
        bit_count_ += 8;

        // Handle JPEG bit stuffing removal
        if (byte == 0xFF && byte_pos_ < buffer_.size()) {
            if (buffer_[byte_pos_] == 0x00) {
                byte_pos_++; // Skip the stuffed zero
            }
        }
    }
    if (byte_pos_ >= buffer_.size() && bit_count_ == 0) {
        eof_flag_ = true;
    }
}

uint32_t BitReader::readBits(const int count) {
    if (count == 0) return 0;

    if (bit_count_ < count) {
        fillBuffer();
    }

    if (bit_count_ < count) {
        // Not enough bits left, this indicates an issue or end of stream
        eof_flag_ = true;
        return 0; // Return 0 for safety
    }

    bit_count_ -= count;
    const uint32_t result = (bit_buffer_ >> bit_count_) & ((1U << count) - 1);
    return result;
}

bool BitReader::isEOF() const {
    return eof_flag_ && bit_count_ == 0;
}
