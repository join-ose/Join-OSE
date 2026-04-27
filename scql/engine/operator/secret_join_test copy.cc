// Copyright 2024 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "engine/operator/secret_join.h"

#include "gtest/gtest.h"

#include "engine/core/tensor_constructor.h"
#include "engine/operator/test_util.h"
#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/type.h>
#include <chrono>
#include <iostream>
#include <random>

namespace scql::engine::op {

struct SecretJoinTestCase {
  std::vector<test::NamedTensor> left_key;
  std::vector<test::NamedTensor> right_key;
  std::vector<test::NamedTensor> left_payload;
  std::vector<test::NamedTensor> right_payload;
  std::vector<test::NamedTensor> expect_left_output;
  std::vector<test::NamedTensor> expect_right_output;
};

class SecretJoinTest
    : public ::testing::TestWithParam<
          std::tuple<test::SpuRuntimeTestCase, SecretJoinTestCase>> {
 protected:
  static pb::ExecNode MakeSecretJoinExecNode(const SecretJoinTestCase& tc);

  static void FeedInputs(const std::vector<ExecContext*>& ctxs,
                         const SecretJoinTestCase& tc);
};

INSTANTIATE_TEST_SUITE_P(
    SecretJoinSecretTest, SecretJoinTest,
    testing::Combine(
        testing::Values(test::SpuRuntimeTestCase{spu::ProtocolKind::SEMI2K, 2},
                        test::SpuRuntimeTestCase{spu::ProtocolKind::SEMI2K, 3}),
        testing::Values(
            SecretJoinTestCase{
                .left_key = {test::NamedTensor(
                    "lk", TensorFrom(arrow::large_utf8(),
                                     R"json(["a", "b", "a", "c", "b"])json"))},
                .right_key = {test::NamedTensor(
                    "rk", TensorFrom(arrow::large_utf8(),
                                     R"json(["a", "a", "b", "d", "a"])json"))},
                .left_payload =
                    {test::NamedTensor(
                         "lp0",
                         TensorFrom(arrow::large_utf8(),
                                    R"json(["a", "b", "a", "c", "b"])json")),
                     test::NamedTensor(
                         "lp1",
                         TensorFrom(
                             arrow::large_utf8(),
                             R"json(["x1", "y1", "x2", "z1", "y2"])json"))},
                .right_payload =
                    {test::NamedTensor(
                         "rp0",
                         TensorFrom(arrow::large_utf8(),
                                    R"json(["a", "a", "b", "d", "a"])json")),
                     test::NamedTensor(
                         "rp1",
                         TensorFrom(
                             arrow::large_utf8(),
                             R"json(["r1", "r2", "s1", "t1", "r3"])json"))},
                .expect_left_output =
                    {test::NamedTensor(
                         "lp0",
                         TensorFrom(
                             arrow::large_utf8(),
                             R"json(["b", "b", "a", "a", "a", "a", "a", "a"])json")),
                     test::NamedTensor(
                         "lp1",
                         TensorFrom(
                             arrow::
                                 large_utf8(),
                             R"json(["y1", "y2", "x1", "x1", "x1", "x2", "x2", "x2"])json"))},
                .expect_right_output =
                    {test::NamedTensor(
                         "rp0",
                         TensorFrom(
                             arrow::large_utf8(),
                             R"json(["b", "b", "a", "a", "a", "a", "a", "a"])json")),
                     test::NamedTensor(
                         "rp1",
                         TensorFrom(
                             arrow::
                                 large_utf8(),
                             R"json(["s1", "s1", "r1", "r2", "r3", "r1", "r2", "r3"])json"))}},
            SecretJoinTestCase{
                .left_key = {test::NamedTensor(
                    "lk", TensorFrom(arrow::int64(), "[1, 2, 1, 2, 0]"))},
                .right_key = {test::NamedTensor(
                    "rk", TensorFrom(arrow::int64(), "[1, 2, 1, 2]"))},
                .left_payload =
                    {test::NamedTensor("lp0", TensorFrom(arrow::int64(),
                                                         "[1, 2, 1, 2, 0]")),
                     test::NamedTensor("lp1", TensorFrom(arrow::int64(),
                                                         "[0, 1, 2, 3, 4]"))},
                .right_payload =
                    {test::NamedTensor("rp0", TensorFrom(arrow::int64(),
                                                         "[1, 2, 1, 2]")),
                     test::NamedTensor("rp1", TensorFrom(arrow::int64(),
                                                         "[0, 1, 2, 3]"))},
                .expect_left_output =
                    {test::NamedTensor("lp0",
                                       TensorFrom(arrow::int64(),
                                                  "[1, 1, 1, 1, 2, 2, 2, 2]")),
                     test::NamedTensor("lp1",
                                       TensorFrom(arrow::int64(),
                                                  "[0, 0, 2, 2, 1, 1, 3, 3]"))},
                .expect_right_output =
                    {test::NamedTensor("rp0",
                                       TensorFrom(arrow::int64(),
                                                  "[1, 1, 1, 1, 2, 2, 2, 2]")),
                     test::NamedTensor(
                         "rp1", TensorFrom(arrow::int64(),
                                           "[0, 2, 0, 2, 1, 3, 1, 3]"))}},
            SecretJoinTestCase{
                .left_key =
                    {test::NamedTensor(
                         "lk0", TensorFrom(arrow::int64(), "[1, 1, 2, 2, 5]")),
                     test::NamedTensor(
                         "lk1",
                         TensorFrom(arrow::large_utf8(),
                                    R"json(["a", "a", "b", "d", "a"])json"))},
                .right_key =
                    {test::NamedTensor("rk0", TensorFrom(arrow::int64(),
                                                         "[1, 1, 2, 2, 4, 6]")),
                     test::NamedTensor(
                         "rk1",
                         TensorFrom(
                             arrow::large_utf8(),
                             R"json(["a", "a", "b", "c", "e", "f"])json"))},
                .left_payload = {test::NamedTensor(
                    "lp1", TensorFrom(arrow::int64(), "[0, 1, 2, 3, 4]"))},
                .right_payload = {test::NamedTensor(
                    "rp1", TensorFrom(arrow::int64(), "[0, 1, 2, 3, 4, 5]"))},
                .expect_left_output = {test::NamedTensor(
                    "lo1", TensorFrom(arrow::int64(), "[0, 0, 1, 1, 2]"))},
                .expect_right_output = {test::NamedTensor(
                    "ro1", TensorFrom(arrow::int64(), "[0, 1, 0, 1, 2]"))}},
            SecretJoinTestCase{
                .left_key =
                    {test::NamedTensor(
                         "lk", TensorFrom(arrow::int64(), "[1, 1, 2, 2, 5]")),
                     test::NamedTensor(
                         "lk0", TensorFrom(arrow::float64(),
                                           "[1.0, 2.2, 3.3, 4.4, -5.5]")),
                     test::NamedTensor(
                         "lk1",
                         TensorFrom(arrow::large_utf8(),
                                    R"json(["a", "a", "b", "d", "a"])json"))},
                .right_key =
                    {test::NamedTensor("rk", TensorFrom(arrow::int64(),
                                                        "[1, 1, 2, 2, 4, 6]")),
                     test::NamedTensor(
                         "rk0",
                         TensorFrom(arrow::float64(),
                                    "[-1.0, -2.2, -3.3, -4.4, -5.5, 6.6]")),
                     test::NamedTensor(
                         "rk1",
                         TensorFrom(
                             arrow::large_utf8(),
                             R"json(["a", "a", "b", "c", "e", "f"])json"))},
                .left_payload = {test::NamedTensor(
                    "lp1", TensorFrom(arrow::int64(), "[0, 1, 2, 3, 4]"))},
                .right_payload = {test::NamedTensor(
                    "rp1", TensorFrom(arrow::int64(), "[0, 1, 2, 3, 4, 5]"))},
                .expect_left_output = {test::NamedTensor(
                    "lo1", TensorFrom(arrow::int64(), "[]"))},
                .expect_right_output = {test::NamedTensor(
                    "ro1", TensorFrom(arrow::int64(), "[]"))}},
            SecretJoinTestCase{.left_key = {test::NamedTensor(
                                   "lk0", TensorFrom(arrow::int64(), "[]"))},
                               .right_key = {test::NamedTensor(
                                   "rk0", TensorFrom(arrow::int64(), "[]"))},
                               .left_payload = {test::NamedTensor(
                                   "lp1", TensorFrom(arrow::int64(), "[]"))},
                               .right_payload = {test::NamedTensor(
                                   "rp1", TensorFrom(arrow::int64(), "[]"))},
                               .expect_left_output = {test::NamedTensor(
                                   "lo1", TensorFrom(arrow::int64(), "[]"))},
                               .expect_right_output = {test::NamedTensor(
                                   "ro1", TensorFrom(arrow::int64(), "[]"))}})),
    TestParamNameGenerator(SecretJoinTest));

// TEST_P(SecretJoinTest, works) {
//   // Given
//   auto parm = GetParam();
//   auto tc = std::get<1>(parm);
//   auto node = MakeSecretJoinExecNode(tc);
//   auto sessions = test::MakeMultiPCSession(std::get<0>(parm));

//   std::vector<ExecContext> exec_ctxs;
//   exec_ctxs.reserve(sessions.size());
//   for (auto& session : sessions) {
//     exec_ctxs.emplace_back(node, session.get());
//   }

//   // feed inputs
//   std::vector<ExecContext*> ctx_ptrs;
//   ctx_ptrs.reserve(exec_ctxs.size());
//   for (auto& exec_ctx : exec_ctxs) {
//     ctx_ptrs.emplace_back(&exec_ctx);
//   }
//   FeedInputs(ctx_ptrs, tc);

//   // When
//   EXPECT_NO_THROW(test::RunAsync<SecretJoin>(ctx_ptrs));

//   // Then check outputs in alice: reveal all secret output to
//   // alice

//   auto expect_output = tc.expect_left_output;
//   expect_output.insert(expect_output.end(), tc.expect_right_output.begin(),
//                        tc.expect_right_output.end());
//   for (const auto& expect_t : expect_output) {
//     TensorPtr out;
//     EXPECT_NO_THROW(out = test::RevealSecret(ctx_ptrs, expect_t.name));
//     // convert hash to string for string tensor in spu
//     if (expect_t.tensor->Type() == pb::PrimitiveDataType::STRING) {
//       out = ctx_ptrs[0]->GetSession()->HashToString(*out);
//     }
//     EXPECT_TRUE(out != nullptr);
//     // compare tensor content
//     EXPECT_TRUE(out->ToArrowChunkedArray()->Equals(
//         expect_t.tensor->ToArrowChunkedArray()))
//         << "actual output = " << out->ToArrowChunkedArray()->ToString()
//         << ", expect output = "
//         << expect_t.tensor->ToArrowChunkedArray()->ToString();
//   }
// }

/// ===========================
/// SecretJoinTest impl
/// ===========================

pb::ExecNode SecretJoinTest::MakeSecretJoinExecNode(
    const SecretJoinTestCase& tc) {
  test::ExecNodeBuilder builder(SecretJoin::kOpType);

  builder.SetNodeName("secret-join-test");
  // Add inputs
  {
    std::vector<pb::Tensor> input_datas;
    for (const auto& named_tensor : tc.left_key) {
      auto data = test::MakeSecretTensorReference(named_tensor.name,
                                                  named_tensor.tensor->Type());
      input_datas.push_back(std::move(data));
    }
    builder.AddInput(SecretJoin::kLeftKey, input_datas);
  }
  {
    std::vector<pb::Tensor> input_datas;
    for (const auto& named_tensor : tc.right_key) {
      auto data = test::MakeSecretTensorReference(named_tensor.name,
                                                  named_tensor.tensor->Type());
      input_datas.push_back(std::move(data));
    }
    builder.AddInput(SecretJoin::kRightKey, input_datas);
  }
  {
    std::vector<pb::Tensor> input_datas;
    for (const auto& named_tensor : tc.left_payload) {
      auto data = test::MakeSecretTensorReference(named_tensor.name,
                                                  named_tensor.tensor->Type());
      input_datas.push_back(std::move(data));
    }
    builder.AddInput(SecretJoin::kLeftPayload, input_datas);
  }
  {
    std::vector<pb::Tensor> input_datas;
    for (const auto& named_tensor : tc.right_payload) {
      auto data = test::MakeSecretTensorReference(named_tensor.name,
                                                  named_tensor.tensor->Type());
      input_datas.push_back(std::move(data));
    }
    builder.AddInput(SecretJoin::kRightPayload, input_datas);
  }

  // Add outputs
  {
    std::vector<pb::Tensor> outputs;
    for (const auto& named_tensor : tc.expect_left_output) {
      auto data = test::MakeSecretTensorReference(named_tensor.name,
                                                  named_tensor.tensor->Type());
      outputs.push_back(std::move(data));
    }
    builder.AddOutput(SecretJoin::kOutLeft, outputs);
  }
  {
    std::vector<pb::Tensor> outputs;
    for (const auto& named_tensor : tc.expect_right_output) {
      auto data = test::MakeSecretTensorReference(named_tensor.name,
                                                  named_tensor.tensor->Type());
      outputs.push_back(std::move(data));
    }
    builder.AddOutput(SecretJoin::kOutRight, outputs);
  }

  return builder.Build();
}

void SecretJoinTest::FeedInputs(const std::vector<ExecContext*>& ctxs,
                                const SecretJoinTestCase& tc) {
  test::FeedInputsAsSecret(ctxs, tc.left_key);
  test::FeedInputsAsSecret(ctxs, tc.right_key);
  test::FeedInputsAsSecret(ctxs, tc.left_payload);
  test::FeedInputsAsSecret(ctxs, tc.right_payload);
}
class SecretJoinPerfTest : public SecretJoinTest {};
TEST_F(SecretJoinPerfTest, LargeInput_2Pow20_NoCheck) {
  constexpr int64_t kSize = 1LL << 8;
  arrow::Int64Builder left_key_builder;
  arrow::Int64Builder right_key_builder;
  arrow::Int64Builder left_payload_builder;
  arrow::Int64Builder right_payload_builder;
    int64_t nx = 1LL << 8;
    int64_t ny = 1LL << 8;
    int64_t n  = nx / 4;
    std::mt19937 rng(12345);
std::uniform_int_distribution<uint32_t> dist(0, (1u << 30) - 1);
    std::vector<uint32_t> k(n);
    for (int64_t i = 0; i < n; ++i) {
         k[i] = static_cast<int64_t>(dist(rng));  // 等价于 np.random.randint
    }
    for (int repeat = 0; repeat < 4; ++repeat) {
        for (int64_t i = 0; i < n; ++i) {
            left_key_builder.Append(k[i]);
            left_payload_builder.Append(i);  // payload 可自定义
        }
    }
    // 前 n：可匹配部分
    for (int64_t i = 0; i < n; ++i) {
        right_key_builder.Append(k[i]);
        right_payload_builder.Append(i);
    }


// 后半部分：不匹配的随机 key
    for (int64_t i = 0; i < ny -  n; ++i) {
    right_key_builder.Append(static_cast<int64_t>(dist(rng)));
    right_payload_builder.Append(i);
    }


//  for (int64_t i = 0; i < kSize; ++i) {
//   ASSERT_TRUE(left_key_builder.Append(i).ok());
//   ASSERT_TRUE(right_key_builder.Append(i).ok());

//   ASSERT_TRUE(left_payload_builder.Append(i).ok());
//   ASSERT_TRUE(right_payload_builder.Append(i).ok());
// }

  std::shared_ptr<arrow::Array> left_key_arr;
  std::shared_ptr<arrow::Array> right_key_arr;
  std::shared_ptr<arrow::Array> left_payload_arr;
  std::shared_ptr<arrow::Array> right_payload_arr;

  ASSERT_TRUE(left_key_builder.Finish(&left_key_arr).ok());
  ASSERT_TRUE(right_key_builder.Finish(&right_key_arr).ok());
  ASSERT_TRUE(left_payload_builder.Finish(&left_payload_arr).ok());
  ASSERT_TRUE(right_payload_builder.Finish(&right_payload_arr).ok());

  auto left_key_tensor = TensorFrom(
      std::make_shared<arrow::ChunkedArray>(left_key_arr));
  auto right_key_tensor = TensorFrom(
      std::make_shared<arrow::ChunkedArray>(right_key_arr));
  auto left_payload_tensor = TensorFrom(
      std::make_shared<arrow::ChunkedArray>(left_payload_arr));
  auto right_payload_tensor = TensorFrom(
      std::make_shared<arrow::ChunkedArray>(right_payload_arr));

  SecretJoinTestCase tc{
      .left_key = {
          test::NamedTensor("lk", left_key_tensor),
      },
      .right_key = {
          test::NamedTensor("rk", right_key_tensor),
      },
      .left_payload = {
          test::NamedTensor("lp", left_payload_tensor),
      },
      .right_payload = {
          test::NamedTensor("rp", right_payload_tensor),
      },
      // 只提供 schema，不校验
      .expect_left_output = {
          test::NamedTensor("out_lp",
              TensorFrom(arrow::int64(), "[]")),
      },
      .expect_right_output = {
          test::NamedTensor("out_rp",
              TensorFrom(arrow::int64(), "[]")),
      },
  };

  auto node = MakeSecretJoinExecNode(tc);

  auto sessions = test::MakeMultiPCSession(
      test::SpuRuntimeTestCase{spu::ProtocolKind::SEMI2K, 2});

  std::vector<ExecContext> exec_ctxs;
  for (auto& s : sessions) {
    exec_ctxs.emplace_back(node, s.get());
  }

  std::vector<ExecContext*> ctx_ptrs;
  for (auto& ctx : exec_ctxs) {
    ctx_ptrs.push_back(&ctx);
  }

  FeedInputs(ctx_ptrs, tc);
  using Clock = std::chrono::steady_clock;

  auto t0 = Clock::now();
  EXPECT_NO_THROW({
    test::RunAsync<SecretJoin>(ctx_ptrs);
  });
  auto t1 = Clock::now();
    auto local_ms =
    std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    GTEST_LOG_(INFO) << "SecretJoin local time(ms): " << local_ms;
}




}  // namespace scql::engine::op
