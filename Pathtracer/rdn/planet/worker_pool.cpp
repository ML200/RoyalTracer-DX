//====================================
//PLANET - WORKER POOL
//====================================

#include "worker_pool.h"
#include <atomic>

namespace planet {

WorkerPool::WorkerPool(uint32_t thread_count) {
    uint32_t n = thread_count;
    if (n == 0) {
        n = std::thread::hardware_concurrency();
        if (n == 0) n = 4;
        if (n > 1)  n -= 1;          // leave one core for the calling thread
    }
    m_threads.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
        m_threads.emplace_back([this] { worker_loop(); });
}

WorkerPool::~WorkerPool() {
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_shutdown = true;
    }
    m_cv.notify_all();
    for (std::thread& t : m_threads)
        if (t.joinable()) t.join();
}

void WorkerPool::enqueue(std::function<void()> job) {
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_jobs.push_back(std::move(job));
    }
    m_cv.notify_one();
}

bool WorkerPool::pop_and_run() {
    std::function<void()> job;
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        if (m_jobs.empty()) return false;
        job = std::move(m_jobs.front());
        m_jobs.pop_front();
    }
    job();
    return true;
}

void WorkerPool::worker_loop() {
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lk(m_mutex);
            m_cv.wait(lk, [&] { return m_shutdown || !m_jobs.empty(); });
            if (m_shutdown && m_jobs.empty()) return;   // drain remaining jobs, then exit
            job = std::move(m_jobs.front());
            m_jobs.pop_front();
        }
        job();
    }
}

void WorkerPool::parallel_for(uint32_t count, const std::function<void(uint32_t)>& body) {
    if (count == 0) return;
    if (m_threads.empty()) {                         // no workers: run inline
        for (uint32_t i = 0; i < count; ++i) body(i);
        return;
    }

    std::atomic<uint32_t> remaining{ count };
    for (uint32_t i = 0; i < count; ++i) {
        enqueue([&body, &remaining, i] {
            body(i);
            remaining.fetch_sub(1, std::memory_order_release);
        });
    }
    //the calling thread helps drain the queue, then waits out any stragglers
    while (remaining.load(std::memory_order_acquire) != 0) {
        if (!pop_and_run())
            std::this_thread::yield();
    }
}

size_t WorkerPool::pending() {
    std::lock_guard<std::mutex> lk(m_mutex);
    return m_jobs.size();
}

} // namespace planet
