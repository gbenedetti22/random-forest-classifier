#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <thread>

namespace pbar {

class ProgressBar {
    // Read-mostly members grouped together
    alignas(64) size_t total_;
    size_t ncols_;
    uint64_t update_interval_ms_{100};
    size_t update_frequency_{1};
    std::chrono::steady_clock::time_point start_time_;

    // Frequently updated by update()
    alignas(64) std::atomic<size_t> current_{0};
    std::atomic<uint64_t> last_update_time_ms_{0};
    std::atomic<bool> completed_{false};

    // Frequently updated by set_description
    alignas(64) std::atomic<bool> prefix_lock_{false};
    std::string prefix_;

    alignas(64) std::mutex console_mutex_; // Only for console output std::cerr

public:
    explicit ProgressBar(size_t total, const std::string& prefix = "", size_t ncols = 50)
        : total_(total), ncols_(ncols), prefix_(prefix) {
        start_time_ = std::chrono::steady_clock::now();
        last_update_time_ms_.store(get_now_ms(), std::memory_order_relaxed);
        prefix_.reserve(128); // prevent allocations during copy
        
        // Heuristic to avoid std::chrono::now() overhead on extremely fast loops.
        // The display is still time-throttled to 100ms, but we only CHECK the time
        // every N iterations based on the total workload.
        if (total_ > 1000000) {
            update_frequency_ = 4096;
        } else if (total_ > 100000) {
            update_frequency_ = 512;
        } else if (total_ > 10000) {
            update_frequency_ = 64;
        } else {
            update_frequency_ = 1;
        }
    }

    // Disable copy/move safely
    ProgressBar(const ProgressBar&) = delete;
    ProgressBar& operator=(const ProgressBar&) = delete;

    ~ProgressBar() {
        finish();
    }

    void update(const size_t increment = 1) {
        // Fast path: accumulate locally to prevent cache line bouncing on the global atomic counter.
        // This is crucial for high-performance loops (e.g. OpenMP parallel for) where multiple threads 
        // incrementing a single atomic millions of times per second would destroy performance.
        thread_local size_t local_count = 0;
        local_count += increment;
        
        if (local_count >= update_frequency_) {
            // Batch update the global atomic counter
            const size_t current = current_.fetch_add(local_count, std::memory_order_relaxed) + local_count;
            local_count = 0; // reset local accumulator
            
            if (current >= total_) {
                // Ensure the final 100% progress is printed exactly once
                if (!completed_.exchange(true, std::memory_order_acq_rel)) {
                    display(total_);
                    std::cerr << std::endl; // newline on finish
                }
                return;
            }

            const uint64_t now_ms = get_now_ms();

            if (uint64_t last = last_update_time_ms_.load(std::memory_order_relaxed); now_ms - last >= update_interval_ms_) {
                // Lock-free check to ensure only ONE thread performs the console output 
                // in the given time window, thus preventing garbled lines.
                if (last_update_time_ms_.compare_exchange_strong(last, now_ms, std::memory_order_relaxed)) {
                    display(current);
                }
            }
        }
    }
    
    void set_description(const std::string& desc) {
        // Optimistic read (Shared cache line) avoids invalidating other cores' caches
        if (prefix_lock_.load(std::memory_order_relaxed)) return;

        if (bool expected = false; prefix_lock_.compare_exchange_strong(expected, true, std::memory_order_acquire, std::memory_order_relaxed)) {
            prefix_ = desc; // capacity is reserved, so no allocation overhead
            prefix_lock_.store(false, std::memory_order_release);
        }
    }

    void set_description(const char* desc) {
        // Overload to prevent implicit std::string construction BEFORE the lock is acquired.
        if (prefix_lock_.load(std::memory_order_relaxed)) return;

        if (bool expected = false; prefix_lock_.compare_exchange_strong(expected, true, std::memory_order_acquire, std::memory_order_relaxed)) {
            prefix_ = desc;
            prefix_lock_.store(false, std::memory_order_release);
        }
    }

    void set_progress(const size_t value) {
        current_.store(value, std::memory_order_relaxed);
        
        if (value >= total_) {
            if (!completed_.exchange(true, std::memory_order_acq_rel)) {
                display(total_);
                std::cerr << std::endl;
            }
            return;
        }

        const uint64_t now_ms = get_now_ms();

        if (uint64_t last = last_update_time_ms_.load(std::memory_order_relaxed); now_ms - last >= update_interval_ms_) {
            if (last_update_time_ms_.compare_exchange_strong(last, now_ms, std::memory_order_relaxed)) {
                display(value);
            }
        }
    }

    // Allow forcing a completion message explicitly
    void finish() {
        if (!completed_.exchange(true, std::memory_order_acq_rel)) {
            display(total_);
            std::cerr << std::endl;
        }
    }

private:
    static uint64_t get_now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    static std::string format_time(std::chrono::seconds seconds) {
        const auto h = std::chrono::duration_cast<std::chrono::hours>(seconds);
        seconds -= h;
        const auto m = std::chrono::duration_cast<std::chrono::minutes>(seconds);
        seconds -= m;
        
        std::ostringstream oss;
        oss << std::setfill('0') << std::setw(2) << h.count() << ":"
            << std::setfill('0') << std::setw(2) << m.count() << ":"
            << std::setfill('0') << std::setw(2) << seconds.count();
        return oss.str();
    }

    void display(size_t current) {
        if (total_ == 0) return;
        if (current > total_) current = total_;

        const float progress = static_cast<float>(current) / static_cast<float>(total_);
        const size_t pos = ncols_ * progress;

        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);

        // Safely extract the current prefix
        bool expected = false;
        while (!prefix_lock_.compare_exchange_strong(expected, true, std::memory_order_acquire, std::memory_order_relaxed)) {
            expected = false;
            // spin wait. This is extremely short because writer holds it for < 5ns.
            std::this_thread::yield();
        }
        const std::string current_prefix = prefix_;
        prefix_lock_.store(false, std::memory_order_release);

        std::lock_guard lock(console_mutex_);

        std::ostringstream out;
        out << "\r" << current_prefix;
        if (!current_prefix.empty()) {
            out << " ";
        }
        
        out << "|";
        for (size_t i = 0; i < ncols_; ++i) {
            if (i < pos) out << "█";
            else if (i == pos) out << "█"; // Can be half-block, but using full block for simplicity
            else out << " ";
        }
        
        // Formatting standard tqdm-like output
        out << "| " << static_cast<int>(progress * 100.0f) << "% ["
            << current << "/" << total_ << " | "
            << format_time(elapsed) << "]";

        // Use a single write to std::cerr to prevent threading issues
        std::cerr << out.str() << std::flush;
    }
};

} // namespace pbar
