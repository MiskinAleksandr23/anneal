#include <algorithm>
#include <atomic>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <mutex>
#include <print>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

using usize = size_t;

namespace sim {
struct query {
  usize left, right;
  usize ts_in, t_start = 0, t_out = 0;
  usize lock_wait = 0;

  query(usize left, usize right, usize ts_in)
      : left(left), right(right), ts_in(ts_in) {}
};

using History = std::vector<query>;

struct MutexDistribution {
  usize n;
  std::vector<usize> starts;
};

usize simulate(usize start, History &hs, const MutexDistribution &md,
               usize thread_count, usize a, usize b) {
  assert(thread_count > 0);
  assert(md.n > 0);
  assert(!md.starts.empty());
  assert(md.starts[0] == 0);
  assert(md.starts.back() < md.n);

  for (usize i = 1; i < md.starts.size(); ++i) {
    assert(md.starts[i - 1] < md.starts[i]);
  }

  std::ranges::stable_sort(hs, std::less{}, &query::ts_in);

  struct Worker {
    query *job = nullptr;
    usize first = 0, next = 0, last = 0;
    usize ready_at = 0, asleep_since = 0;
    bool asleep = false;
  };

  struct SimMutex {
    bool busy = false;
    std::deque<usize> waiters;
  };

  std::vector<Worker> workers(thread_count);
  std::vector<SimMutex> locks(md.starts.size());

  auto get_block = [&md](usize position) -> usize {
    assert(position < md.n);
    usize i = 0;

    while (i + 1 < md.starts.size() && md.starts[i + 1] <= position)
      ++i;
    return i;
  };

  usize now = start, total_response = 0;
  usize pending = 0, completed = 0;

  auto grant_access = [&](Worker &worker) {
    locks[worker.next++].busy = true;
    worker.ready_at = now + b;

    if (worker.next == worker.last + 1) {
      worker.ready_at += (worker.job->right - worker.job->left + 1) * a;
    }
  };

  while (completed < hs.size()) {
    usize selected = workers.size();
    usize event_time = 0;

    for (usize i = 0; i < workers.size(); ++i) {
      const auto &worker = workers[i];
      if (worker.asleep || (!worker.job && pending == hs.size()))
        continue;
      const usize tm =
          worker.job ? worker.ready_at : std::max(now, hs[pending].ts_in);
      if (selected == workers.size() || tm < event_time) {
        selected = i;
        event_time = tm;
      }
    }
    assert(selected < workers.size());
    now = event_time;

    auto &worker = workers[selected];
    if (!worker.job) {
      worker.job = &hs[pending++];
      assert(worker.job->left <= worker.job->right);
      assert(worker.job->right < md.n);
      worker.job->t_start = now;
      worker.job->lock_wait = 0;
      worker.first = get_block(worker.job->left);
      worker.next = worker.first;
      worker.last = get_block(worker.job->right);
    }

    if (worker.next <= worker.last) {
      auto &lock = locks[worker.next];
      if (lock.busy) {
        lock.waiters.push_back(selected);
        worker.asleep = true;
        worker.asleep_since = now;
      } else {
        grant_access(worker);
      }
      continue;
    }

    worker.job->t_out = now;
    total_response += now - worker.job->ts_in;

    for (usize i = worker.first; i <= worker.last; ++i) {
      auto &lock = locks[i];
      lock.busy = false;
      if (!lock.waiters.empty()) {
        auto &waiter = workers[lock.waiters.front()];
        lock.waiters.pop_front();

        waiter.job->lock_wait += now - waiter.asleep_since;
        waiter.asleep = false;
        grant_access(waiter);
      }
    }
    worker.job = nullptr;
    ++completed;
  }

  return total_response;
}

}; // namespace sim

namespace {
static constexpr usize kWorkers = 8;
static constexpr usize kA = 1;
static constexpr usize kB = 3;
static constexpr usize kBlocksCount = 9;
static constexpr usize kN = 1 << 14;
static constexpr usize kSeed = 42;
static constexpr usize kMeanArrivalInterval = 50;
} // namespace

usize score(sim::History &hs, const sim::MutexDistribution &md, usize workers) {
  return sim::simulate(0, hs, md, workers, kA, kB);
};

sim::MutexDistribution search(sim::History &train, sim::MutexDistribution p,
                              usize workers) {
  auto best = score(train, p, workers);
  bool improved = false;
  usize evaluations = 1;
  usize pass = 0;

  do {
    improved = false;
    for (usize i = 1; i < p.starts.size(); ++i) {
      const auto original = p.starts[i];
      auto best_cut = original;
      const auto limit = i + 1 < p.starts.size() ? p.starts[i + 1] : p.n;
      for (auto cut = p.starts[i - 1] + 1; cut < limit; ++cut) {
        if (cut == original)
          continue;
        p.starts[i] = cut;
        const auto cost = score(train, p, workers);
        ++evaluations;
        if (cost < best) {
          best = cost;
          best_cut = cut;
        }
      }
      p.starts[i] = best_cut;
      improved |= best_cut != original;
    }
    std::println("search pass={} score={} evaluations={}", ++pass, best,
                 evaluations);
  } while (improved);
  return p;
}

sim::MutexDistribution anneal(sim::History &train,
                              sim::MutexDistribution current, usize workers) {
  constexpr usize kSteps = 4'000;
  constexpr usize kLogInterval = 100;
  constexpr double kInitialTemperatureRatio = 0.005;
  constexpr double kFinalTemperatureRatio = 0.01;
  constexpr unsigned kSeed = 121345;

  if (train.empty() || current.starts.size() < 2)
    return current;

  auto current_cost = score(train, current, workers);
  auto best_cost = current_cost;
  auto best = current;

  const usize radius =
      std::max<usize>(1, current.n / current.starts.size() / 4);
  double temperature = std::max(1.0, kInitialTemperatureRatio * current_cost);
  const double cooling = std::pow(kFinalTemperatureRatio, 1.0 / (kSteps - 1));
  std::mt19937 rng(kSeed);
  std::uniform_int_distribution<usize> pick_cut(1, current.starts.size() - 1);
  std::uniform_real_distribution<double> probability(0.0, 1.0);

  auto try_move = [&]() {
    const auto i = pick_cut(rng);
    const auto original = current.starts[i];
    const auto limit =
        i + 1 < current.starts.size() ? current.starts[i + 1] : current.n;
    const auto left =
        original - std::min(radius, original - current.starts[i - 1] - 1);
    const auto right = original + std::min(radius, limit - original - 1);
    if (left == right)
      return;

    auto cut = std::uniform_int_distribution<usize>(left, right - 1)(rng);
    if (cut >= original)
      ++cut;
    current.starts[i] = cut;

    const auto cost = score(train, current, workers);
    if (cost > current_cost) {
      const double delta = static_cast<double>(cost - current_cost);
      if (probability(rng) >= std::exp(-delta / temperature)) {
        current.starts[i] = original;
        return;
      }
    }

    current_cost = cost;
    if (cost < best_cost) {
      best_cost = cost;
      best = current;
    }
  };

  usize last_log_step = 0;
  for (usize step = 1; step <= kSteps; ++step) {
    const auto previous_best = best_cost;
    try_move();
    if (best_cost < previous_best && step - last_log_step >= kLogInterval) {
      std::println("anneal step={}/{} best={}", step, kSteps, best_cost);
      last_log_step = step;
    }
    temperature *= cooling;
  }
  return best;
}

sim::History make_history_v1(usize seed, usize mean_arrival_interval) {
  constexpr usize kBursts = 1'000;
  constexpr usize kBurstSize = 8;
  assert(mean_arrival_interval > 0);
  const usize burst_period = kBurstSize * mean_arrival_interval;
  constexpr usize kCentersPct[] = {5,  9,  14, 23, 31, 44,
                                   50, 61, 69, 78, 87, 95};

  std::mt19937 rng(seed);
  std::discrete_distribution<usize> pick_center(
      {6, 4, 5, 2, 4, 3, 6, 2, 5, 3, 2, 4});
  auto draw = [&](usize lo, usize hi) {
    return std::uniform_int_distribution<usize>(lo, hi)(rng);
  };

  sim::History history;
  history.reserve(kBursts * kBurstSize);
  for (usize burst = 0; burst < kBursts; ++burst) {
    const usize center = kN * kCentersPct[pick_center(rng)] / 100;
    assert(center >= 96 && kN - center >= 96);
    for (usize j = 0; j < kBurstSize; ++j) {
      usize left, right;
      if (j == kBurstSize - 1 && burst % 8 == 0) {
        left = center - draw(32, 96);
        right = center + draw(32, 96) - 1;
      } else if (j % 2 == 0) {
        right = center - 1 - draw(0, 24);
        left = right + 1 - draw(16, 64);
      } else {
        left = center + draw(0, 24);
        right = left + draw(16, 64) - 1;
      }
      history.emplace_back(left, right, burst * burst_period + draw(0, 5));
    }
  }
  return history;
}

sim::History make_history_v2(usize seed, usize mean_arrival_interval) {
  assert(mean_arrival_interval > 0);
  constexpr auto kMaxTime = std::numeric_limits<usize>::max();
  if (mean_arrival_interval - 1 > kMaxTime - mean_arrival_interval)
    throw std::overflow_error("Arrival interval is too large");

  auto history = make_history_v1(seed, 1);
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<usize> gap(1, mean_arrival_interval +
                                                  (mean_arrival_interval - 1));

  history.front().ts_in = 0;
  for (usize i = 1; i < history.size(); ++i) {
    const auto interval = gap(rng);
    const auto previous = history[i - 1].ts_in;
    if (interval > kMaxTime - previous)
      throw std::overflow_error("Arrival time is too large");
    history[i].ts_in = previous + interval;
  }
  return history;
}

sim::MutexDistribution make_uniform(usize size, usize blocks_count) {
  assert(blocks_count >= 1 && size >= blocks_count);
  sim::MutexDistribution result{size, {}};
  result.starts.reserve(blocks_count);

  const usize width = size / blocks_count;
  const usize remainder = size % blocks_count;
  usize start = 0;
  for (usize i = 0; i < blocks_count; ++i) {
    result.starts.push_back(start);
    start += width + (i < remainder ? 1 : 0);
  }
  return result;
}

double mean_service_time(const sim::History &history,
                         const sim::MutexDistribution &md) {
  if (history.empty())
    return 0.0;

  usize total = 0;
  for (const auto &q : history) {
    const auto locks = 1 + std::ranges::count_if(md.starts, [&](usize cut) {
                         return q.left < cut && cut <= q.right;
                       });
    total += (q.right - q.left + 1) * kA + locks * kB;
  }
  return static_cast<double>(total) / history.size();
}

void example() {
  auto history = make_history_v2(kSeed, kMeanArrivalInterval);
  auto uniform = make_uniform(kN, kBlocksCount);
  auto uniform_score = score(history, uniform, kWorkers);

  std::println("seed={} uniform_score={}", kSeed, uniform_score);
  auto md = anneal(history, uniform, kWorkers);
  auto anneal_score = score(history, md, kWorkers);

  std::println("uniform_score= {}\nanneal_score = {}", uniform_score,
               anneal_score);
  if (!history.empty()) {
    const auto count = static_cast<double>(history.size());
    std::println(
        "uniform_mean_response = {:.3f}  uniform_mean_service = {:.3f}",
        uniform_score / count, mean_service_time(history, uniform));
    std::println("anneal_mean_response = {:.3f}  anneal_mean_service = {:.3f}",
                 anneal_score / count, mean_service_time(history, md));
  }
  for (usize i = 0; i < md.starts.size(); ++i) {
    const auto end = i + 1 < md.starts.size() ? md.starts[i + 1] : md.n;
    std::println("mutex {}: [{}, {}] ({} cells)", i, md.starts[i], end - 1,
                 end - md.starts[i]);
  }
}

template <typename Mutex = std::mutex> class FixedLock {
  struct alignas(128) Slot {
    Mutex mutex;
  };

  std::vector<usize> block_of_;
  std::vector<Slot> locks_;

public:
  explicit FixedLock(const sim::MutexDistribution &md)
      : block_of_(md.n), locks_(md.starts.size()) {
    for (usize i = 0; i < md.starts.size(); ++i) {
      const auto end = i + 1 < md.starts.size() ? md.starts[i + 1] : md.n;
      std::fill(block_of_.begin() + md.starts[i], block_of_.begin() + end, i);
    }
  }

  void lock(usize left, usize right) {
    for (usize i = block_of_[left]; i <= block_of_[right]; ++i)
      locks_[i].mutex.lock();
  }

  void unlock(usize left, usize right) {
    for (usize i = block_of_[right] + 1; i > block_of_[left]; --i)
      locks_[i - 1].mutex.unlock();
  }
};

#if defined(__aarch64__)
#include <arm_acle.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

inline void spin_pause() noexcept {
#if defined(__aarch64__)
  __yield();
#elif defined(__x86_64__) || defined(__i386__)
  _mm_pause();
#endif
}

#define RANDOM_BACKOFF

#ifdef NAIVE
struct Spinlock {
  void lock() noexcept {
    while (locked_.load(std::memory_order_relaxed) ||
           locked_.exchange(true, std::memory_order_acquire)) {
    }
  }
  void unlock() noexcept { locked_.store(false, std::memory_order_release); }

  std::atomic<bool> locked_ = false;
};
#elifdef EXP_BACKOFF
#define MIN_BACKOFF 4
#define MAX_BACKOFF 1024
class Spinlock {
private:
  // Lock is just an atomic bool
  std::atomic<bool> locked{false};

public:
  // Locking mechanism
  void lock() {
    // Start backoff at MIN_BACKOFF iterations
    int backoff_iters = MIN_BACKOFF;

    // Keep trying
    while (1) {
      // Try and grab the lock
      // Return if we get the lock
      if (!locked.exchange(true, std::memory_order_acquire))
        return;

      // If we didn't get the lock, just read the value which gets cached
      // locally. This leads to less traffic.
      // Pause for an exponentially increasing number of iterations
      do {
        // Pause for some number of iterations
        for (int i = 0; i < backoff_iters; i++)
          spin_pause();

        // Get the backoff iterations for next time
        backoff_iters = std::min(backoff_iters << 1, MAX_BACKOFF);

        // Check to see if the lock is free
      } while (locked.load(std::memory_order_relaxed));
    }
  }

  // Unlocking mechanism
  // Just set the lock to free (false)
  // Can also use the assignment operator
  void unlock() { locked.store(false, std::memory_order_release); }
};
#elifdef RANDOM_BACKOFF
class Spinlock {
private:
  // Lock is just an atomic bool
  std::atomic<bool> locked{false};

public:
  // Locking mechanism
  void lock() {
    // Keep trying
    while (1) {
      // Try and grab the lock
      // Exit if we got the lock
      if (!locked.exchange(true, std::memory_order_acquire))
        return;

      // If we didn't get the lock, just read the value which gets cached
      // locally. This leads to less traffic.
      // Pause for a random number of iterations (between 4 and 1024)
      do {
        // Get a random number of iterations between 4 and 1024
        static thread_local std::mt19937 rng{std::random_device{}()};
        const int backoff_iters =
            std::uniform_int_distribution<int>(4, 1024)(rng);

        // Pause for some number of iterations
        for (int i = 0; i < backoff_iters; i++)
          spin_pause();

        // Check to see if the lock is now free
      } while (locked.load(std::memory_order_relaxed));
    }
  }

  // Unlocking mechanism
  // Just set the lock to free (false)
  // Can also use the assignment operator
  void unlock() { locked.store(false, std::memory_order_release); }
};
#endif

template <typename Mutex = std::mutex>
double bench_distribution(const sim::History &hs,
                          const sim::MutexDistribution &md) {
  constexpr usize kFull = 10'000;
  using Clock = std::chrono::steady_clock;
  const auto task_count = kFull * hs.size();
  if (task_count == 0)
    return 0.0;

  FixedLock<Mutex> locks(md);
  std::vector<usize> data(md.n);
  std::atomic<usize> next{0};
  std::vector<Clock::duration> elapsed(kWorkers);
  std::barrier start(kWorkers);
  std::vector<std::thread> threads;
  for (usize tid = 0; tid < kWorkers; ++tid) {
    threads.emplace_back([&, tid] {
      start.arrive_and_wait();
      const auto begin = Clock::now();
      for (;;) {
        const auto ticket = next.fetch_add(1, std::memory_order_relaxed);
        if (ticket >= task_count)
          break;
        const auto &q = hs[ticket % hs.size()];
        locks.lock(q.left, q.right);
        for (usize i = q.left; i <= q.right; ++i)
          ++data[i];
        locks.unlock(q.left, q.right);
      }
      elapsed[tid] = Clock::now() - begin;
    });
  }
  for (auto &thread : threads)
    thread.join();

  Clock::duration total{};
  for (auto time : elapsed)
    total += time;
  return std::chrono::duration<double, std::nano>(total).count() / task_count;
}

void bench() {
  auto history = make_history_v2(kSeed, kMeanArrivalInterval);
  auto uniform = make_uniform(kN, kBlocksCount);
  auto md = anneal(history, uniform, kWorkers);

  auto baseline = bench_distribution(history, uniform);
  auto fast = bench_distribution(history, md);
  std::println(
      "uniform_mean_ns_per_op = {:.2f}\nanneal_mean_ns_per_op = {:.2f}",
      baseline, fast);
}

int main() { bench(); }