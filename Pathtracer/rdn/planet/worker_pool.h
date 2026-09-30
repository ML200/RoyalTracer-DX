#pragma once

#include <cstdint>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace planet {

class WorkerPool {
public:
    explicit WorkerPool(uint32_t thread_count = 0);
    ~WorkerPool();

    WorkerPool(const WorkerPool&)            = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    void enqueue(std::function<void()> job);

    // Blocks; the caller helps drain the queue.
    void parallel_for(uint32_t count, const std::function<void(uint32_t)>& body);

    uint32_t thread_count() const { return (uint32_t)m_threads.size(); }

private:
    void worker_loop();
    bool pop_and_run();

    std::vector<std::thread>          m_threads;
    std::deque<std::function<void()>> m_jobs;
    std::mutex                        m_mutex;
    std::condition_variable           m_cv;
    bool                              m_shutdown = false;
};

}
