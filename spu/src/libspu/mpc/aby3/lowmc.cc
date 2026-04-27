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

#include "libspu/mpc/aby3/lowmc.h"

#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/aby3/type.h"
#include "libspu/mpc/utils/lowmc.h"
#include "libspu/mpc/utils/lowmc_utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::aby3 {

namespace {

FieldType get_field_from_pt_type(PtType pt_type) {
  size_t size = SizeOf(pt_type);
  if (size <= 4) return FM32;
  if (size <= 8) return FM64;
  return FM128;
}

PtType get_pt_type_from_field(FieldType field) {
  switch (field) {
    case FM32: return PT_U32;
    case FM64: return PT_U64;
    case FM128: return PT_U128;
    default: SPU_THROW("unsupported field type");
  }
}

// =====================================================================
// SPU 内置 MPC 算子封装
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

// 针对 BShrTy 的安全 GF2 矩阵乘法
NdArrayRef dot_product_gf2_bshr(const NdArrayRef& matrix, const NdArrayRef& state, FieldType to_field) {
  const auto back_type = state.eltype().as<BShrTy>()->getBacktype();
  const auto ret_bty = makeType<BShrTy>(get_pt_type_from_field(to_field), SizeOf(to_field) * 8);
  auto ret = NdArrayRef(ret_bty, state.shape());

  DISPATCH_UINT_PT_TYPES(back_type, [&]() {
    using bshr_t = std::array<ScalarT, 2>;
    NdArrayView<bshr_t> _state(state);
    
    const auto field = get_field_from_pt_type(back_type);
    NdArrayRef s0(makeType<RingTy>(field), state.shape());
    NdArrayRef s1(makeType<RingTy>(field), state.shape());
    
    DISPATCH_ALL_FIELDS(field, [&]() {
      using ring_t = ring2k_t;
      NdArrayView<ring_t> _s0(s0);
      NdArrayView<ring_t> _s1(s1);
      pforeach(0, state.numel(), [&](int64_t idx) {
        _s0[idx] = static_cast<ring_t>(_state[idx][0]);
        _s1[idx] = static_cast<ring_t>(_state[idx][1]);
      });
    });

    auto ret0 = dot_product_gf2(matrix, s0, to_field);
    auto ret1 = dot_product_gf2(matrix, s1, to_field);

    DISPATCH_ALL_FIELDS(to_field, [&]() {
      using dst_ring_t = ring2k_t;
      NdArrayView<dst_ring_t> _ret0(ret0);
      NdArrayView<dst_ring_t> _ret1(ret1);
      
      DISPATCH_UINT_PT_TYPES(get_pt_type_from_field(to_field), [&]() {
        using dst_bshr_t = std::array<ScalarT, 2>;
        NdArrayView<dst_bshr_t> _ret(ret);
        pforeach(0, state.numel(), [&](int64_t idx) {
          _ret[idx][0] = static_cast<ScalarT>(_ret0[idx]);
          _ret[idx][1] = static_cast<ScalarT>(_ret1[idx]);
        });
      });
    });
  });
  return ret;
}

NdArrayRef Sbox(KernelEvalContext* ctx, const NdArrayRef& state,
                int64_t n_boxes, size_t n_bits) {
  const auto back_type = state.eltype().as<BShrTy>()->getBacktype();
  const auto field = get_field_from_pt_type(back_type);
  const auto shape = state.shape();
  auto sctx = ctx->sctx(); // 获取 SPUContext

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

  // 修改点：使用 sctx 替代 ctx
  auto S_M0 = wrap_and_bp(sctx, state, M0); 
  auto S_M1 = wrap_and_bp(sctx, state, M1); 
  auto S_M2 = wrap_and_bp(sctx, state, M2); 

  auto Y_c = wrap_lshift_b(sctx, S_M0, 1);
  auto Y_b = wrap_lshift_b(sctx, S_M1, 1);
  auto Y_a = wrap_rshift_b(sctx, S_M2, 2);
  auto Y = wrap_xor_bb(sctx, wrap_xor_bb(sctx, Y_c, Y_b), Y_a);

  auto X = wrap_and_bp(sctx, state, M210);
  auto Z = wrap_and_bb(sctx, X, Y);

  auto Z_M0 = wrap_and_bp(sctx, Z, M0); 
  auto Z_M1 = wrap_and_bp(sctx, Z, M1); 
  auto Z_M2 = wrap_and_bp(sctx, Z, M2); 

  auto bc_to_a = wrap_lshift_b(sctx, Z_M1, 1);
  auto new_a = wrap_xor_bb(sctx, S_M2, bc_to_a);

  auto ca_to_b = wrap_lshift_b(sctx, Z_M0, 1);
  auto a_to_b = wrap_rshift_b(sctx, S_M2, 1);
  auto new_b = wrap_xor_bb(sctx, wrap_xor_bb(sctx, S_M1, a_to_b), ca_to_b);

  auto ab_to_c = wrap_rshift_b(sctx, Z_M2, 2);
  auto a_to_c = wrap_rshift_b(sctx, S_M2, 2);
  auto b_to_c = wrap_rshift_b(sctx, S_M1, 1);
  auto new_c = wrap_xor_bb(sctx, wrap_xor_bb(sctx, wrap_xor_bb(sctx, S_M0, a_to_c), b_to_c), ab_to_c);

  auto new_sbox = wrap_xor_bb(sctx, wrap_xor_bb(sctx, new_a, new_b), new_c);
  auto S_rest = wrap_and_bp(sctx, state, M_rest);
  
  return wrap_xor_bb(sctx, new_sbox, S_rest);
}

NdArrayRef Affine(KernelEvalContext* ctx, const LowMC& cipher,
                  const NdArrayRef& state, int64_t rounds) {
  const auto back_type = state.eltype().as<BShrTy>()->getBacktype();
  const auto field = get_field_from_pt_type(back_type);
  const auto L_matrix = cipher.Lmat()[rounds];
  return dot_product_gf2_bshr(L_matrix, state, field);
}

}  // namespace

NdArrayRef LowMcB::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* prg_state = ctx->getState<PrgState>();

  uint128_t key0 = 0;
  uint128_t key1 = 0;
  prg_state->fillPrssPair(&key0, &key1, 1, PrgState::GenPrssCtrl::Both);

  uint128_t seed;
  prg_state->fillPubl(absl::MakeSpan(&seed, 1));

  return encrypt(ctx, in, key0, key1, seed);
}

NdArrayRef LowMcB::encrypt(KernelEvalContext* ctx, const NdArrayRef& in,
                           uint128_t key0, uint128_t key1, uint128_t seed) const {
  const auto back_type = in.eltype().as<BShrTy>()->getBacktype();
  const auto field = get_field_from_pt_type(back_type);
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

    auto rk0 = generate_round_keys(cipher.Kmat(), key0, cipher.rounds(), field);
    auto rk1 = generate_round_keys(cipher.Kmat(), key1, cipher.rounds(), field);

    const auto bshr_ty = makeType<BShrTy>(back_type, k);

    auto build_bshr_rk = [&](int round) {
      NdArrayRef res(bshr_ty, shape);
      DISPATCH_UINT_PT_TYPES(back_type, [&]() {
        using bshr_el_t = ScalarT;
        using bshr_t = std::array<bshr_el_t, 2>;
        NdArrayView<bshr_t> _res(res);
        
        NdArrayView<ring2k_t> _rk0(rk0[round]);
        NdArrayView<ring2k_t> _rk1(rk1[round]);
        auto val0 = static_cast<ScalarT>(_rk0[0]);
        auto val1 = static_cast<ScalarT>(_rk1[0]);
        
        pforeach(0, shape.numel(), [&](int64_t idx) {
          _res[idx][0] = val0;
          _res[idx][1] = val1;
        });
      });
      return res;
    };

    auto rk0_bshr = build_bshr_rk(0);
    out = wrap_xor_bb(sctx, in, rk0_bshr);

    const auto n_boxes = cipher.number_of_boxes();
    for (int64_t r = 1; r <= cipher.rounds(); ++r) {
      out = Sbox(ctx, out, n_boxes, k);
      out = Affine(ctx, cipher, out, r - 1);

      auto round_constant = cipher.RoundConstants()[r - 1].broadcast_to(shape, {}).as(pub_ty);
      out = wrap_xor_bp(sctx, out, round_constant);

      auto rk_r_bshr = build_bshr_rk(r);
      out = wrap_xor_bb(sctx, out, rk_r_bshr);
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
  const auto in_nbits = inputs[0].eltype().as<BShrTy>()->nbits();
  const auto k = in_nbits;

  auto ret_bty = makeType<BShrTy>(get_pt_type_from_field(dst_field), inputs.size() * k);
  auto ret = NdArrayRef(ret_bty, inputs[0].shape());

  DISPATCH_UINT_PT_TYPES(get_pt_type_from_field(dst_field), [&]() {
    using dst_t = std::array<ScalarT, 2>;
    NdArrayView<dst_t> _ret(ret);

    pforeach(0, ret.numel(), [&](int64_t idx) {
      _ret[idx][0] = 0;
      _ret[idx][1] = 0;
    });

    for (uint64_t i = 0; i < inputs.size(); ++i) {
      const auto tmp_back_type = inputs[i].eltype().as<BShrTy>()->getBacktype();
      DISPATCH_UINT_PT_TYPES(tmp_back_type, [&]() {
        using src_t = std::array<ScalarT, 2>;
        NdArrayView<src_t> _inp(inputs[i]);
        const auto shift_bits = k * i;

        pforeach(0, ret.numel(), [&](int64_t idx) {
          _ret[idx][0] |= (static_cast<ScalarT>(_inp[idx][0]) << shift_bits);
          _ret[idx][1] |= (static_cast<ScalarT>(_inp[idx][1]) << shift_bits);
        });
      });
    }
  });

  return ret;
}
}  // namespace

NdArrayRef MultiKeyLowMcB::proc(KernelEvalContext* ctx,
                                const std::vector<NdArrayRef>& inputs) const {
  SPU_ENFORCE(!inputs.empty());
  const auto back_type = inputs[0].eltype().as<BShrTy>()->getBacktype();
  const auto field = get_field_from_pt_type(back_type);
  
  if (inputs.size() == 1) {
    return wrap_lowmcb(ctx, inputs[0]);
  }

  static constexpr int64_t kMaxBits = 128;
  static constexpr FieldType kMaxField = FM128;
  const int64_t k = inputs[0].eltype().as<BShrTy>()->nbits();
  const auto total_bits = k * inputs.size();

  if (total_bits <= kMaxBits) {
    const auto dst_field = get_dst_field(total_bits);
    auto concat_inp = concate_bits(inputs, dst_field);
    return wrap_lowmcb(ctx, concat_inp);
  } else {
    auto* prg_state = ctx->getState<PrgState>();
    const Shape rand_mat_shape = {kMaxBits};
    auto remapping_inp_ty = makeType<BShrTy>(get_pt_type_from_field(kMaxField), kMaxBits);
    auto remapping_inp = NdArrayRef(remapping_inp_ty, inputs[0].shape());

    DISPATCH_UINT_PT_TYPES(get_pt_type_from_field(kMaxField), [&]() {
      using dst_t = std::array<ScalarT, 2>;
      NdArrayView<dst_t> _remap(remapping_inp);
      pforeach(0, remapping_inp.numel(), [&](int64_t idx) {
        _remap[idx][0] = 0;
        _remap[idx][1] = 0;
      });
    });

    for (const auto& item : inputs) {
      const auto rand_mat = prg_state->genPubl(field, rand_mat_shape);
      auto part_dot = dot_product_gf2_bshr(rand_mat, item, kMaxField);
      
      DISPATCH_UINT_PT_TYPES(get_pt_type_from_field(kMaxField), [&]() {
        using dst_t = std::array<ScalarT, 2>;
        NdArrayView<dst_t> _remap(remapping_inp);
        const auto tmp_back_type = part_dot.eltype().as<BShrTy>()->getBacktype();
        DISPATCH_UINT_PT_TYPES(tmp_back_type, [&]() {
          using src_t = std::array<ScalarT, 2>;
          NdArrayView<src_t> _part(part_dot);
          pforeach(0, remapping_inp.numel(), [&](int64_t idx) {
            _remap[idx][0] ^= static_cast<ScalarT>(_part[idx][0]);
            _remap[idx][1] ^= static_cast<ScalarT>(_part[idx][1]);
          });
        });
      });
    }
    return wrap_lowmcb(ctx, remapping_inp);
  }
}

}  // namespace spu::mpc::aby3