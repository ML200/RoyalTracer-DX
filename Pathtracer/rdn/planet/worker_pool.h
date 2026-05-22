#pragma once
//====================================
//PLANET - WORKER POOL
//====================================
//Job-queue thread pool for CPU chunk tessellation.
//  enqueue()      - fire-and-forget; the job runs on some worker thread later.
//                   This is the Phase 4 async tessellation path.
//  parallel_for() - blocking; runs body(0..count-1) across the workers (the
//                   calling thread joins in) and returns once all have finished.
//enqueue() is thread-safe and may be called from any thread.

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
    explicit WorkerPool(uint32_t thread_count = 0);   // 0 -> hardware_concurrency() - 1
    ~WorkerPool();

    WorkerPool(const WorkerPool&)            = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    //fire-and-forget: 'job' runs on a worker thread at some later point
    void enqueue(std::function<void()> job);

    //run body(i) for every i in [0, count); blocks until all have completed
    void parallel_for(uint32_t count, const std::function<void(uint32_t)>& body);

    uint32_t thread_count() const { return (uint32_t)m_threads.size(); }
    size_t   pending();                               // queued jobs not yet started

private:
    void worker_loop();
    bool pop_and_run();                               // pop+run one job; false if queue empty

    std::vector<std::thread>          m_threads;
    std::deque<std::function<void()>> m_jobs;
    std::mutex                        m_mutex;
    std::condition_variable           m_cv;
    bool                              m_shutdown = false;
};

} // namespace planet
