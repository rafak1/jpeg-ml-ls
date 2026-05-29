#ifndef BIT_STREAM_HPP
#define BIT_STREAM_HPP

#include <vector>
#include <cstdint>

// BitWriter to handle writing bits to a byte buffer
class BitWriter {
public:
    void writeBits(uint32_t bits, int count);
    void flush();
    [[nodiscard]] const std::vector<uint8_t>& getBuffer() const;
private:
    std::vector<uint8_t> buffer_;
    uint32_t bit_buffer_ = 0;
    int bit_count_ = 0;
};

// BitReader to handle reading bits from a byte buffer
class BitReader {
public:
    explicit BitReader(const std::vector<uint8_t>& buffer);
    uint32_t readBits(int count);
    [[nodiscard]] bool isEOF() const;
private:
    const std::vector<uint8_t>& buffer_;
    size_t byte_pos_ = 0;
    uint32_t bit_buffer_ = 0;
    int bit_count_ = 0;
    bool eof_flag_ = false;

    void fillBuffer();
};

#endif // BIT_STREAM_HPP
