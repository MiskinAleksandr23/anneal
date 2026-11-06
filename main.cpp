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
static constexpr usize kMaxTiming = 20'000;
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
#ifdef DEBUG_SEARCH_OUTPUT
    std::println("search pass={} score={} evaluations={}", ++pass, best,
                 evaluations);
#else
    ++pass;
#endif
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
#ifdef DEBUG_SEARCH_OUTPUT
      std::println("anneal step={}/{} best={}", step, kSteps, best_cost);
#endif
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

sim::History make_history_v3(usize seed, usize mean_arrival_interval) {
  assert(mean_arrival_interval > 0);
  constexpr usize kQueries = 8'000;
  constexpr double kMeanQueryLength = 48.0;

  std::mt19937_64 rng(seed);
  std::exponential_distribution<double> arrival_gap(
      1.0 / static_cast<double>(mean_arrival_interval));
  std::geometric_distribution<usize> query_length(1.0 / kMeanQueryLength);

  sim::History history;
  history.reserve(kQueries);
  usize now = 0;
  for (usize i = 0; i < kQueries; ++i) {
    now += std::max<usize>(1, static_cast<usize>(arrival_gap(rng)));
    const usize length = std::min<usize>(query_length(rng), kN - 1) + 1;
    const usize left =
        std::uniform_int_distribution<usize>(0, kN - length)(rng);
    history.emplace_back(left, left + length - 1, now);
  }
  return history;
}

sim::History make_history_v4(usize seed, usize mean_arrival_interval) {
  assert(mean_arrival_interval > 0);
  constexpr usize kQueries = 8'000;
  constexpr int kMaxWalkStep = 128;

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> walk_step(-kMaxWalkStep, kMaxWalkStep);
  std::uniform_int_distribution<usize> length_exponent(0, 12);
  std::normal_distribution<double> local_offset(0.0,
                                                static_cast<double>(kN) / 64.0);
  std::bernoulli_distribution global_query(0.1);

  sim::History history;
  history.reserve(kQueries);
  long long center = kN / 2;
  for (usize i = 0; i < kQueries; ++i) {
    center = std::clamp(center + walk_step(rng), 0LL,
                        static_cast<long long>(kN - 1));
    const usize length = usize{1} << length_exponent(rng);

    long long left;
    if (global_query(rng)) {
      left = std::uniform_int_distribution<usize>(0, kN - length)(rng);
    } else {
      const long long max_left = static_cast<long long>(kN - length);
      const long long proposed = center + std::llround(local_offset(rng)) -
                                 static_cast<long long>(length) / 2;
      left = std::clamp(proposed, 0LL, max_left);
    }
    history.emplace_back(static_cast<usize>(left),
                         static_cast<usize>(left) + length - 1,
                         i * mean_arrival_interval);
  }
  return history;
}

sim::History make_history_v5(usize seed, usize mean_arrival_interval) {
  assert(mean_arrival_interval > 0);
  constexpr usize kQueries = 8'192;
  constexpr usize kPeriod = 512;
  constexpr double kTwoPi = 6.28318530717958647692;

  std::mt19937_64 rng(seed);
  std::discrete_distribution<usize> pick_level(
      {1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1});

  sim::History history;
  history.reserve(kQueries);
  usize now = 0;
  for (usize i = 0; i < kQueries; ++i) {
    if (i != 0) {
      const double phase =
          static_cast<double>(i % kPeriod) / static_cast<double>(kPeriod);
      const double gap_factor = 1.0 + 0.8 * std::sin(kTwoPi * phase);
      const auto gap = static_cast<usize>(std::llround(
          static_cast<double>(mean_arrival_interval) * gap_factor));
      now += std::max<usize>(1, gap);
    }

    const usize length = usize{1} << pick_level(rng);
    const usize block =
        std::uniform_int_distribution<usize>(0, kN / length - 1)(rng);
    const usize left = block * length;
    history.emplace_back(left, left + length - 1, now);
  }
  return history;
}

sim::History make_history_v6(usize seed) {
  constexpr usize kClients = 64;
  constexpr usize kRounds = 128;
  static_assert(kN % kClients == 0);
  constexpr usize kRegionWidth = kN / kClients;

  std::mt19937_64 rng(seed);
  std::gamma_distribution<double> think_time(2.0, 30.0);
  std::uniform_int_distribution<usize> local_length(1, 64);

  sim::History history;
  history.reserve(kClients * kRounds);
  std::vector<usize> ready_at(kClients, 0);
  for (usize round = 0; round < kRounds; ++round) {
    for (usize client = 0; client < kClients; ++client) {
      const usize ts_in = ready_at[client];
      if (round % 16 == 0 && client % 8 == 0) {
        history.emplace_back(0, kN - 1, ts_in);
      } else {
        const usize region_left = client * kRegionWidth;
        const usize length = local_length(rng);
        const usize left = std::uniform_int_distribution<usize>(
            region_left, region_left + kRegionWidth - length)(rng);
        history.emplace_back(left, left + length - 1, ts_in);
      }
      ready_at[client] +=
          std::max<usize>(1, static_cast<usize>(std::llround(think_time(rng))));
    }
  }
  return history;
}

sim::History make_history_v7(usize seed) {
  constexpr usize kWaves = 512;
  constexpr usize kLanes = 16;
  static_assert(kN % kLanes == 0);
  constexpr usize kLaneWidth = kN / kLanes;

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<usize> lane_query_length(64, 256);

  sim::History history;
  history.reserve(kWaves * (kLanes + 1));
  for (usize wave = 0; wave < kWaves; ++wave) {
    const usize base = wave * 128;
    history.emplace_back(0, kN - 1, base + wave % 2);
    for (usize lane = 0; lane < kLanes; ++lane) {
      const usize lane_left = lane * kLaneWidth;
      const usize length = lane_query_length(rng);
      const usize left = std::uniform_int_distribution<usize>(
          lane_left, lane_left + kLaneWidth - length)(rng);
      history.emplace_back(left, left + length - 1, base + 1 - wave % 2);
    }
  }
  return history;
}

sim::History make_history_v8(usize seed) {
  constexpr usize kScans = 64;
  constexpr usize kSteps = 128;
  constexpr usize kStepInterval = 4;

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<usize> window_exponent(6, 9);

  sim::History history;
  history.reserve(kScans * kSteps);
  for (usize scan = 0; scan < kScans; ++scan) {
    const usize window = usize{1} << window_exponent(rng);
    const usize max_left = kN - window;
    for (usize step = 0; step < kSteps; ++step) {
      const usize progress = scan % 2 == 0 ? step : kSteps - step - 1;
      const usize left = max_left * progress / (kSteps - 1);
      history.emplace_back(left, left + window - 1,
                           (scan * kSteps + step) * kStepInterval);
    }
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
void bench_distribution(const char *name, const sim::History &hs,
                        const sim::MutexDistribution &md) {
  constexpr usize kFull = 10'000;
  using Clock = std::chrono::steady_clock;
  const auto task_count = kFull * hs.size();
  if (task_count == 0)
    return;

  FixedLock<Mutex> locks(md);
  std::vector<usize> data(md.n);
  std::atomic<usize> next{0};
  std::vector<Clock::duration> elapsed(kWorkers);
  std::vector<std::vector<usize>> counts(kWorkers,
                                         std::vector<usize>(kMaxTiming + 1));
  std::vector<usize> over_limits(kWorkers);
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
        const auto op_begin = Clock::now();
        locks.lock(q.left, q.right);
        for (usize i = q.left; i <= q.right; ++i)
          ++data[i];
        locks.unlock(q.left, q.right);
        const auto timing = std::max<usize>(
            1, static_cast<usize>(
                   std::chrono::duration_cast<std::chrono::nanoseconds>(
                       Clock::now() - op_begin)
                       .count()));
        if (timing <= kMaxTiming)
          ++counts[tid][timing];
        else
          ++over_limits[tid];
      }
      elapsed[tid] = Clock::now() - begin;
    });
  }
  for (auto &thread : threads)
    thread.join();

  Clock::duration total{};
  std::vector<usize> total_counts(kMaxTiming + 1);
  usize over_limit = 0;
  for (usize tid = 0; tid < kWorkers; ++tid) {
    total += elapsed[tid];
    over_limit += over_limits[tid];
    for (usize timing = 1; timing <= kMaxTiming; ++timing)
      total_counts[timing] += counts[tid][timing];
  }

  std::println("{}_mean_ns_per_op = {:.2f}", name,
               std::chrono::duration<double, std::nano>(total).count() /
                   task_count);
  auto print_percentile = [&](usize percent) {
    const auto target = static_cast<usize>(
        std::ceil(static_cast<double>(task_count) * percent / 100.0));
    usize accumulated = 0;
    for (usize timing = 1; timing <= kMaxTiming; ++timing) {
      accumulated += total_counts[timing];
      if (accumulated >= target) {
        std::println("{}_p{}_ns = {}", name, percent, timing);
        return;
      }
    }
    std::println("{}_p{}_ns = >{}", name, percent, kMaxTiming);
  };
  print_percentile(50);
  print_percentile(75);
  print_percentile(90);
  print_percentile(95);
  print_percentile(99);
  std::println("{}_over_max = {}", name, over_limit);
}

void bench_history(usize version, const char *description,
                   sim::History history) {
  std::println("\nmake_history_v{}: {}", version, description);
  auto uniform = make_uniform(kN, kBlocksCount);
  auto md = anneal(history, uniform, kWorkers);

  bench_distribution("uniform", history, uniform);
  bench_distribution("anneal", history, md);
}

void bench() {
  bench_history(1, "bursts around weighted hot spots",
                make_history_v1(kSeed, kMeanArrivalInterval));
  bench_history(2, "hot-spot queries with uniform interarrival gaps",
                make_history_v2(kSeed, kMeanArrivalInterval));
  bench_history(3, "Poisson arrivals and geometric query lengths",
                make_history_v3(kSeed, kMeanArrivalInterval));
  bench_history(4, "random-walk hot spot with local and global queries",
                make_history_v4(kSeed, kMeanArrivalInterval));
  bench_history(5, "sinusoidal arrival gaps and power-of-two aligned queries",
                make_history_v5(kSeed, kMeanArrivalInterval));
  bench_history(6, "partitioned clients with periodic global queries",
                make_history_v6(kSeed));
  bench_history(7, "barrier waves with global and per-lane queries",
                make_history_v7(kSeed));
  bench_history(8, "alternating forward and backward range scans",
                make_history_v8(kSeed));
}

int main() { bench(); }
