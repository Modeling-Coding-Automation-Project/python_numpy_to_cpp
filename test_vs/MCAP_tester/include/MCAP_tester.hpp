#ifndef MCAP_TESTER_HPP
#define MCAP_TESTER_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Tester {

template <typename T> class MCAPTester {
public:
  MCAPTester() : test_failed_flag(false) {}

  void expect_near(T actual, T expected, T tolerance,
                   const std::string &message) {
    if (std::abs(actual - expected) <= tolerance) {
      /* Do Nothing. */
    } else {
      std::cout << "FAILURE: " << message << std::endl;
      std::cout << std::endl;
      test_failed_flag = true;
    }
  }

  void expect_near(std::vector<T> actual, std::vector<T> expected, T tolerance,
                   const std::string &message) {
    if (actual.size() != expected.size()) {
      std::cout << "FAILURE: " << message << " Size mismatch." << std::endl;
      std::cout << std::endl;
      test_failed_flag = true;
      return;
    }

    for (size_t i = 0; i < actual.size(); i++) {
      if (std::abs(actual[i] - expected[i]) <= tolerance) {
        /* Do Nothing. */
      } else {
        std::cout << "FAILURE: " << message << " Element mismatch."
                  << std::endl;
        std::cout << std::endl;
        test_failed_flag = true;
        return;
      }
    }
  }

  void expect_near(std::vector<std::vector<T>> actual,
                   std::vector<std::vector<T>> expected, T tolerance,
                   const std::string &message) {
    if (actual.size() != expected.size()) {
      std::cout << "FAILURE: " << message << " Size mismatch." << std::endl;
      std::cout << std::endl;
      test_failed_flag = true;
      return;
    }

    for (size_t i = 0; i < actual.size(); i++) {
      if (actual[i].size() != expected[i].size()) {
        std::cout << "FAILURE: " << message << " Size mismatch. " << std::endl;
        std::cout << std::endl;
        test_failed_flag = true;
        return;
      }

      for (size_t j = 0; j < actual[i].size(); j++) {
        if (std::abs(actual[i][j] - expected[i][j]) <= tolerance) {
          /* Do Nothing. */
        } else {
          std::cout << "FAILURE: " << message << " Element mismatch."
                    << std::endl;
          std::cout << std::endl;
          test_failed_flag = true;
          return;
        }
      }
    }
  }

  void throw_error_if_test_failed() {
    if (test_failed_flag) {
      throw std::runtime_error("Test failed.");
    }
  }

  void reset_test_failed_flag() { test_failed_flag = false; }

private:
  bool test_failed_flag = false;
};

} // namespace Tester

#endif // MCAP_TESTER_HPP
