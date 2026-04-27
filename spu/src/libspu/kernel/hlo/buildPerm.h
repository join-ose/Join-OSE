
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/geometrical.h"
#include "libspu/kernel/hlo/soprf.h"
#include "libspu/kernel/hlo/indexing.h"

namespace spu {
class SPUContext;
}
namespace spu::kernel::hlo {
std::vector<spu::Value> buildPerm(spu::SPUContext* sctx, const spu::Value& p,
                     const spu::Value& pi);
}