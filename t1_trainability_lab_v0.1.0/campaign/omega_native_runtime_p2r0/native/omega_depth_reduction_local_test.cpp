#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr size_t kDimension = 128;
constexpr size_t kHiddenDimension = 4 * kDimension;
constexpr float kAbsoluteTolerance = 1.0e-5F;
constexpr float kRelativeTolerance = 1.0e-4F;

struct Contribution {
  size_t round;
  std::vector<float> upstream;
};

struct Result {
  std::vector<float> production;
  std::vector<float> baseline;
  std::vector<double> oracle;
};

std::vector<float> make_weights() {
  std::vector<float> weights(kHiddenDimension * kDimension);
  for (size_t row = 0; row < kHiddenDimension; ++row) {
    for (size_t input = 0; input < kDimension; ++input) {
      const float sign = ((row + input) & 1U) == 0U ? 1.0F : -1.0F;
      weights[row * kDimension + input] = sign * (0.001F +
          static_cast<float>((row * 17 + input * 7) % 97) * 0.0001F);
    }
  }
  return weights;
}

Result reduce(size_t rounds, const std::vector<float>& weights, const std::vector<Contribution>& contributions) {
  std::vector<float> local(rounds * kHiddenDimension, 0.0F);
  Result result{std::vector<float>(rounds * kDimension, 0.0F),
      std::vector<float>(rounds * kDimension, 0.0F),
      std::vector<double>(rounds * kDimension, 0.0)};

  for (const Contribution& contribution : contributions) {
    float* local_round = local.data() + contribution.round * kHiddenDimension;
    for (size_t row = 0; row < kHiddenDimension; ++row) {
      local_round[row] += contribution.upstream[row];
    }
    for (size_t input = 0; input < kDimension; ++input) {
      const size_t output = contribution.round * kDimension + input;
      for (size_t row = 0; row < kHiddenDimension; ++row) {
        const float product = weights[row * kDimension + input] * contribution.upstream[row];
        result.baseline[output] += product;
        result.oracle[output] += static_cast<double>(weights[row * kDimension + input]) *
            static_cast<double>(contribution.upstream[row]);
      }
    }
  }

  for (size_t round = 0; round < rounds; ++round) {
    const float* local_round = local.data() + round * kHiddenDimension;
    for (size_t input = 0; input < kDimension; ++input) {
      const size_t output = round * kDimension + input;
      for (size_t row = 0; row < kHiddenDimension; ++row) {
        result.production[output] += weights[row * kDimension + input] * local_round[row];
      }
    }
  }
  return result;
}

bool close(float candidate, double oracle) {
  const double error = std::fabs(static_cast<double>(candidate) - oracle);
  return error <= static_cast<double>(kAbsoluteTolerance) +
      static_cast<double>(kRelativeTolerance) * std::fabs(oracle);
}

void require(bool condition, const char* case_name, const char* detail) {
  if (!condition) {
    std::fprintf(stderr, "FAIL case=%s detail=%s\n", case_name, detail);
    std::abort();
  }
}

void run_case(const char* case_name, size_t rounds, const std::vector<Contribution>& contributions) {
  const Result result = reduce(rounds, make_weights(), contributions);
  double max_oracle_error = 0.0;
  double max_baseline_error = 0.0;
  for (size_t index = 0; index < result.production.size(); ++index) {
    const double oracle_error = std::fabs(static_cast<double>(result.production[index]) - result.oracle[index]);
    const double baseline_error = std::fabs(static_cast<double>(result.production[index]) -
        static_cast<double>(result.baseline[index]));
    max_oracle_error = std::max(max_oracle_error, oracle_error);
    max_baseline_error = std::max(max_baseline_error, baseline_error);
    require(close(result.production[index], result.oracle[index]), case_name, "FP64 tolerance");
    require(std::fabs(result.production[index] - result.baseline[index]) <=
        kAbsoluteTolerance + kRelativeTolerance * std::fabs(result.baseline[index]), case_name, "baseline tolerance");
  }
  std::printf("PASS case=%s rounds=%zu contributions=%zu max_oracle_error=%.9g max_baseline_error=%.9g\n",
      case_name, rounds, contributions.size(), max_oracle_error, max_baseline_error);
}

std::vector<float> values(float scale, size_t seed) {
  std::vector<float> result(kHiddenDimension);
  for (size_t row = 0; row < kHiddenDimension; ++row) {
    const float sign = ((row + seed) & 1U) == 0U ? 1.0F : -1.0F;
    result[row] = sign * scale * (0.25F + static_cast<float>((row * 13 + seed * 5) % 101) * 0.01F);
  }
  return result;
}

void run_round_separation_case() {
  const size_t rounds = 4;
  const std::vector<float> weights = make_weights();
  std::vector<Contribution> all;
  for (size_t round = 0; round < rounds; ++round) {
    all.push_back(Contribution{round, values(0.01F * static_cast<float>(round + 1), round)});
    all.push_back(Contribution{round, values(-0.003F * static_cast<float>(round + 2), round + 11)});
  }
  const Result combined = reduce(rounds, weights, all);
  for (size_t round = 0; round < rounds; ++round) {
    std::vector<Contribution> isolated;
    for (const Contribution& contribution : all) {
      if (contribution.round == round) isolated.push_back(contribution);
    }
    const Result one_round = reduce(rounds, weights, isolated);
    for (size_t input = 0; input < kDimension; ++input) {
      const size_t index = round * kDimension + input;
      require(combined.production[index] == one_round.production[index], "round-separation", "round changed");
      for (size_t other = 0; other < rounds; ++other) {
        if (other == round) continue;
        require(one_round.production[other * kDimension + input] == 0.0F,
            "round-separation", "cross-round contribution");
      }
    }
  }
  std::printf("PASS case=round-separation rounds=4 contributions=%zu\n", all.size());
}

}  // namespace

int main() {
  std::mt19937 generator(230);
  std::uniform_real_distribution<float> distribution(-0.75F, 0.75F);

  for (size_t rounds : {size_t{1}, size_t{4}}) {
    std::vector<Contribution> mixed;
    for (size_t index = 0; index < (rounds == 1 ? 9 : 23); ++index) {
      std::vector<float> upstream(kHiddenDimension);
      for (float& value : upstream) value = distribution(generator);
      mixed.push_back(Contribution{index % rounds, std::move(upstream)});
    }
    run_case(rounds == 1 ? "K1-mixed-signs" : "K4-mixed-signs", rounds, mixed);
  }

  std::vector<Contribution> cancellation;
  std::vector<float> positive = values(0.5F, 3);
  std::vector<float> negative = positive;
  for (float& value : negative) value = -value;
  cancellation.push_back(Contribution{0, positive});
  cancellation.push_back(Contribution{0, negative});
  run_case("K1-cancellation", 1, cancellation);

  run_round_separation_case();
  std::printf("PASS step=1-local-depth-reduction\n");
  return 0;
}
