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

#include "libspu/mpc/semi2k/lowmc.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/lowmc.h"
#include "libspu/mpc/utils/lowmc_utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::semi2k {

namespace {

// =====================================================================
// SPU 内置 MPC 算子封装 (利用引擎向量化能力)
// =====================================================================
NdArrayRef wrap_xor_bp(SPUContext* ctx, const NdArrayRef& x, const NdArrayRef& y) {
  return UnwrapValue(xor_bp(ctx, WrapValue(x), WrapValue(y)));
}

NdArrayRef wrap_xor_bb(SPUContext* ctx, const NdArrayRef& x, const NdArrayRef& y) {
  return UnwrapValue(xor_bb(ctx, WrapValue(x), WrapValue(y)));
}

NdArrayRef wrap_and_bb(SPUContext* ctx, const NdArrayRef& x, const NdArrayRef& y) {
  return UnwrapValue(and_bb(ctx, WrapValue(x), WrapValue(y)));
}

NdArrayRef wrap_and_bp(SPUContext* ctx, const NdArrayRef& x, const NdArrayRef& y) {
  return UnwrapValue(and_bp(ctx, WrapValue(x), WrapValue(y)));
}

NdArrayRef wrap_lshift_b(SPUContext* ctx, const NdArrayRef& x, int64_t bits) {
  return UnwrapValue(lshift_b(ctx, WrapValue(x), {bits}));
}

NdArrayRef wrap_rshift_b(SPUContext* ctx, const NdArrayRef& x, int64_t bits) {
  return UnwrapValue(rshift_b(ctx, WrapValue(x), {bits}));
}

// =====================================================================
// 核心优化：比特切片（Bit-Slicing）S 盒
// 彻底干掉 extract_bit_arr 循环，利用掩码和移位批量计算
// =====================================================================
NdArrayRef Sbox(KernelEvalContext* ctx, const NdArrayRef& state,
                int64_t n_boxes, size_t n_bits) {
  const auto field = state.eltype().as<BShrTy>()->field();
  const auto shape = state.shape();
  auto sctx = ctx->sctx();

  // 构造掩码工具
  auto make_mask = [&](uint128_t m_val) {
    auto ret = ring_zeros(field, shape);
    DISPATCH_ALL_FIELDS(field, [&]() {
      NdArrayView<ring2k_t> _ret(ret);
      pforeach(0, shape.numel(), [&](int64_t idx) {
        _ret[idx] = static_cast<ring2k_t>(m_val);
      });
    });
    return ret.as(makeType<Pub2kTy>(field));
  };

  // 生成 c位(0,3,6), b位(1,4,7), a位(2,5,8) 的批量掩码
  uint128_t m0 = 0, m1 = 0, m2 = 0, m_rest = 0;
  for (int64_t i = 0; i < n_boxes; ++i) {
    m0 |= ((uint128_t)1 << (3 * i + 0)); 
    m1 |= ((uint128_t)1 << (3 * i + 1)); 
    m2 |= ((uint128_t)1 << (3 * i + 2)); 
  }
  uint128_t m210 = m0 | m1 | m2; 
  for (size_t i = 3 * n_boxes; i < n_bits; ++i) {
    m_rest |= ((uint128_t)1 << i);
  }

  auto M0 = make_mask(m0);
  auto M1 = make_mask(m1);
  auto M2 = make_mask(m2);
  auto M210 = make_mask(m210);
  auto M_rest = make_mask(m_rest);

  // 1. 提取所有 block 的 a, b, c 比特平面
  auto S_M0 = wrap_and_bp(sctx, state, M0); // bits (c)
  auto S_M1 = wrap_and_bp(sctx, state, M1); // bits (b)
  auto S_M2 = wrap_and_bp(sctx, state, M2); // bits (a)

  // 2. 构造错位交叉项计算乘积 (bc, ac, ab)
  // c -> b, b -> a, a -> c
  auto Y_c = wrap_lshift_b(sctx, S_M0, 1);
  auto Y_b = wrap_lshift_b(sctx, S_M1, 1);
  auto Y_a = wrap_rshift_b(sctx, S_M2, 2);
  auto Y = wrap_xor_bb(sctx, wrap_xor_bb(sctx, Y_c, Y_b), Y_a);

  auto X = wrap_and_bp(sctx, state, M210);
  
  // 这是唯一的通信同步点：一次性计算所有 block 的 3 种乘积项
  auto Z = wrap_and_bb(sctx, X, Y);

  // 3. 分离计算出的乘积平面
  auto Z_M0 = wrap_and_bp(sctx, Z, M0); // ac 平面
  auto Z_M1 = wrap_and_bp(sctx, Z, M1); // bc 平面
  auto Z_M2 = wrap_and_bp(sctx, Z, M2); // ab 平面

  // 4. 根据 LowMC S-box 代数方程重组比特
  // new_a = a ^ bc
  auto bc_to_a = wrap_lshift_b(sctx, Z_M1, 1);
  auto new_a = wrap_xor_bb(sctx, S_M2, bc_to_a);

  // new_b = a ^ b ^ ac
  auto ca_to_b = wrap_lshift_b(sctx, Z_M0, 1);
  auto a_to_b = wrap_rshift_b(sctx, S_M2, 1);
  auto new_b = wrap_xor_bb(sctx, wrap_xor_bb(sctx, S_M1, a_to_b), ca_to_b);

  // new_c = a ^ b ^ c ^ ab
  auto ab_to_c = wrap_rshift_b(sctx, Z_M2, 2);
  auto a_to_c = wrap_rshift_b(sctx, S_M2, 2);
  auto b_to_c = wrap_rshift_b(sctx, S_M1, 1);
  auto new_c = wrap_xor_bb(sctx, wrap_xor_bb(sctx, wrap_xor_bb(sctx, S_M0, a_to_c), b_to_c), ab_to_c);

  // 5. 合并回状态并保留透传高位
  auto new_sbox = wrap_xor_bb(sctx, wrap_xor_bb(sctx, new_a, new_b), new_c);
  auto S_rest = wrap_and_bp(sctx, state, M_rest);
  
  return wrap_xor_bb(sctx, new_sbox, S_rest);
}

NdArrayRef Affine(KernelEvalContext* ctx, const LowMC& cipher,
                  const NdArrayRef& state, int64_t rounds) {
  const auto field = state.eltype().as<BShrTy>()->field();
  const auto L_matrix = cipher.Lmat()[rounds];
  return dot_product_gf2(L_matrix, state, field);
}

}  // namespace

NdArrayRef LowMcB::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* prg_state = ctx->getState<PrgState>();

  uint128_t key;
  prg_state->fillPriv(absl::MakeSpan(&key, 1));

  uint128_t seed;
  prg_state->fillPubl(absl::MakeSpan(&seed, 1));

  return encrypt(ctx, in, key, seed);
}

NdArrayRef LowMcB::encrypt(KernelEvalContext* ctx, const NdArrayRef& in,
                           uint128_t key, uint128_t seed) const {
  const auto field = in.eltype().as<BShrTy>()->field();
  const auto numel = in.numel();
  const auto k = SizeOf(field) * 8;
  const auto shape = in.shape();
  const auto pub_ty = makeType<Pub2kTy>(field);
  auto sctx = ctx->sctx();

  NdArrayRef out;
  DISPATCH_ALL_FIELDS(field, [&]() {
    auto d = get_data_complexity(numel);
    auto cipher = LowMC(field, seed, d);
    SPU_ENFORCE(static_cast<int64_t>(k) == cipher.data_block_size());

    auto round_keys =
        generate_round_keys(cipher.Kmat(), key, cipher.rounds(), field);

    auto round_key0 = round_keys[0].broadcast_to(shape, {}).as(pub_ty);
    out = wrap_xor_bb(sctx, in, round_key0);

    const auto n_boxes = cipher.number_of_boxes();
    for (int64_t r = 1; r <= cipher.rounds(); ++r) {
      out = Sbox(ctx, out, n_boxes, k);
      out = Affine(ctx, cipher, out, r - 1).as(in.eltype());

      auto round_constant =
          cipher.RoundConstants()[r - 1].broadcast_to(shape, {}).as(pub_ty);
      out = wrap_xor_bp(sctx, out, round_constant);

      auto round_key = round_keys[r].broadcast_to(shape, {}).as(pub_ty);
      out = wrap_xor_bb(sctx, out, round_key);
    }
  });

  return out;
}

namespace {
NdArrayRef wrap_lowmcb(KernelEvalContext* ctx, const NdArrayRef& in) {
  return LowMcB().proc(ctx, in);
}

FieldType get_dst_field(const int64_t k) {
  if (k <= 32) return FM32;
  if (k <= 64) return FM64;
  return FM128;
}

NdArrayRef concate_bits(const std::vector<NdArrayRef>& inputs,
                        const FieldType dst_field) {
  const auto field = inputs[0].eltype().as<Ring2k>()->field();
  const auto k = SizeOf(field) * 8;
  auto ret = ring_zeros(dst_field, inputs[0].shape());

  DISPATCH_ALL_FIELDS(field, [&]() {
    using src_el_t = ring2k_t;
    DISPATCH_ALL_FIELDS(dst_field, [&]() {
      using dst_el_t = ring2k_t;
      NdArrayView<dst_el_t> _ret(ret);
      for (uint64_t i = 0; i < inputs.size(); ++i) {
        NdArrayView<src_el_t> _inp(inputs[i]);
        const auto shift_bits = k * i;
        pforeach(0, ret.numel(), [&](int64_t idx) {
          _ret[idx] |= (static_cast<dst_el_t>(_inp[idx]) << shift_bits);
        });
      }
    });
  });
  return ret;
}

}  // namespace

NdArrayRef MultiKeyLowMcB::proc(KernelEvalContext* ctx,
                                const std::vector<NdArrayRef>& inputs) const {
  SPU_ENFORCE(!inputs.empty());
  const auto field = inputs[0].eltype().as<Ring2k>()->field();

  if (inputs.size() == 1) {
    return wrap_lowmcb(ctx, inputs[0]);
  }

  static constexpr int64_t kMaxBits = 128;
  static constexpr FieldType kMaxField = FM128;
  const int64_t k = SizeOf(field) * 8;
  const auto total_bits = k * inputs.size();

  if (total_bits <= kMaxBits) {
    const auto dst_field = get_dst_field(total_bits);
    auto concat_inp = concate_bits(inputs, dst_field).as(makeType<BShrTy>(dst_field));
    return wrap_lowmcb(ctx, concat_inp);
  } else {
    auto* prg_state = ctx->getState<PrgState>();
    const Shape rand_mat_shape = {kMaxBits};
    auto remapping_inp = ring_zeros(kMaxField, inputs[0].shape());
    for (const auto& item : inputs) {
      const auto rand_mat = prg_state->genPubl(field, rand_mat_shape);
      auto part_dot = dot_product_gf2(rand_mat, item, kMaxField);
      ring_xor_(remapping_inp, part_dot);
    }
    return wrap_lowmcb(ctx, remapping_inp.as(makeType<BShrTy>(kMaxField)));
  }
}

}  // namespace spu::mpc::semi2k