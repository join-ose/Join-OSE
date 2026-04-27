#include "libspu/kernel/hlo/buildPerm.h"
#include "absl/container/flat_hash_set.h"
using namespace spu::kernel;
namespace spu::kernel::hlo {
std::vector<spu::Value> buildPerm(spu::SPUContext* sctx, const spu::Value& p,
                                  const spu::Value& pi) {
  auto tmp = hlo::Concatenate(sctx, {pi, p}, 0);
  auto soprf = hlo::SoPrf(sctx, tmp);
  // reveal after soprf
  auto soprf_revealed = hlo::Reveal(sctx, soprf);
  auto view = spu::NdArrayView<uint64_t>(soprf_revealed.data());

  const int64_t pi_numel = pi.numel();
  const int64_t p_numel = p.numel();
  const int64_t total_numel = view.numel();

  // 优化点 1 & 2: 使用 absl::flat_hash_set 替代 std::set，并预分配内存
  absl::flat_hash_set<uint64_t> p_set;
  p_set.reserve(p_numel); 

  // 提取 p 的 SoPrf 结果加入集合
  for (int64_t i = pi_numel; i < total_numel; ++i) {
    p_set.insert(view[i]);
  }

  // 优化点 3: 为 index 预分配最大可能需要的内存，避免扩容
  spu::Index index;
  index.reserve(pi_numel);

  // 提取不在 p_set 中的 pi 元素的索引
  for (int64_t i = 0; i < pi_numel; ++i) {
    if (!p_set.contains(view[i])) { // contains 比 find() == end() 更直观且在 absl 中有优化
      index.push_back(i);
    }
  }

  std::vector<spu::Value> ret;
  ret.emplace_back(hlo::Concatenate(sctx, {p, hlo::LinearGather(sctx, tmp, index)}, 0).setDtype(p.dtype()));
  return ret;
}
// std::vector<spu::Value> buildPerm(spu::SPUContext* sctx, const spu::Value& p,
//                      const spu::Value& pi) {
//   // auto tmp = hlo::Concatenate(sctx, {pi, p}, 0);
//   auto soprf = hlo::SoPrf(sctx, p);
//   // reveal after soprf
//   // auto soprf_revealed = hlo::Reveal(sctx, soprf);
//   // auto view = spu::NdArrayView<uint64_t>(soprf_revealed.data());
//   // std::set<uint64_t> p_set;
//   // for (int64_t i = pi.numel(); i < view.numel(); ++i) {
//   //   p_set.insert(view[i]);
//   // }
//   // spu::Index index;
//   // for (int64_t i = 0; i < pi.numel(); ++i) {
//   //   if (p_set.find(view[i]) == p_set.end()) {
//   //     index.push_back(i);
//   //   }
//   // }
//   std::vector<spu::Value> ret;
//   ret.emplace_back(soprf.setDtype(p.dtype()));
//   // ret.emplace_back(hlo::Concatenate(sctx, {p, hlo::LinearGather(sctx, tmp, index)}, 0).setDtype(p.dtype()));
//   return ret;
// }
// std::vector<spu::Value> buildPerm(spu::SPUContext* sctx, const spu::Value& p,
//                      const spu::Value& pi) {
//   auto tmp = hlo::Concatenate(sctx, {pi, p}, 0);
//   auto soprf = hlo::SoPrf(sctx, tmp);
//   // reveal after soprf
//   auto soprf_revealed = hlo::Reveal(sctx, soprf);
//   auto view = spu::NdArrayView<uint64_t>(soprf_revealed.data());
//   std::set<uint64_t> p_set;
//   for (int64_t i = pi.numel(); i < view.numel(); ++i) {
//     p_set.insert(view[i]);
//   }
//   spu::Index index;
//   for (int64_t i = 0; i < pi.numel(); ++i) {
//     if (p_set.find(view[i]) == p_set.end()) {
//       index.push_back(i);
//     }
//   }
//   std::vector<spu::Value> ret;
//   ret.emplace_back(hlo::Concatenate(sctx, {p, hlo::LinearGather(sctx, tmp, index)}, 0).setDtype(p.dtype()));
//   return ret;
// }
}