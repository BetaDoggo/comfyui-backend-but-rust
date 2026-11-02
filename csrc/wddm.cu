// Windows: per-process GPU memory budget from DXCore (what WDDM lets this process keep resident
// before it starts paging to shared system memory). Host-only code. Loaded dynamically so the
// binary still runs where DXCore is unavailable.
#include <windows.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// {78EE5945-C36E-4B13-A669-005DD11C0F06}
const GUID kIID_IDXCoreAdapterFactory = {0x78ee5945, 0xc36e, 0x4b13, {0xa6, 0x69, 0x00, 0x5d, 0xd1, 0x1c, 0x0f, 0x06}};
// {F0DB4C7F-FE5A-42A2-BD62-F2A6CF6FC83E}
const GUID kIID_IDXCoreAdapter = {0xf0db4c7f, 0xfe5a, 0x42a2, {0xbd, 0x62, 0xf2, 0xa6, 0xcf, 0x6f, 0xc8, 0x3e}};

struct MemoryBudgetQuery {
    uint32_t node_index;
    uint32_t segment_group;  // 0 = local (VRAM)
};
struct MemoryBudget {
    uint64_t budget, current_usage, available_for_reservation, current_reservation;
};

// Only the vtable slots used here are typed; the rest keep their ABI positions.
struct AdapterVtbl {
    void* query_interface;
    void* add_ref;
    ULONG(STDMETHODCALLTYPE* release)(void*);
    void* is_valid;
    void* is_attribute_supported;
    void* is_property_supported;
    void* get_property;
    void* get_property_size;
    void* is_query_state_supported;
    HRESULT(STDMETHODCALLTYPE* query_state)(void*, uint32_t state, size_t in_size, const void* in, size_t out_size,
                                            void* out);
};
struct FactoryVtbl {
    void* query_interface;
    void* add_ref;
    ULONG(STDMETHODCALLTYPE* release)(void*);
    void* create_adapter_list;
    HRESULT(STDMETHODCALLTYPE* get_adapter_by_luid)(void*, const LUID*, REFIID, void**);
};
struct ComObj {
    void* vtbl;
};

void* g_adapter = nullptr;
bool g_tried = false;

bool init() {
    if (g_tried) return g_adapter != nullptr;
    g_tried = true;
    cudaDeviceProp prop;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
    LUID luid;
    memcpy(&luid, prop.luid, sizeof(luid));
    HMODULE lib = LoadLibraryA("dxcore.dll");
    if (!lib) return false;
    typedef HRESULT(WINAPI * CreateFactory)(REFIID, void**);
    auto create = (CreateFactory)GetProcAddress(lib, "DXCoreCreateAdapterFactory");
    if (!create) return false;
    void* factory = nullptr;
    if (FAILED(create(kIID_IDXCoreAdapterFactory, &factory)) || !factory) return false;
    auto fv = (FactoryVtbl*)((ComObj*)factory)->vtbl;
    void* adapter = nullptr;
    HRESULT hr = fv->get_adapter_by_luid(factory, &luid, kIID_IDXCoreAdapter, &adapter);
    fv->release(factory);
    if (FAILED(hr) || !adapter) return false;
    g_adapter = adapter;
    return true;
}

}  // namespace

// 0 on success: *budget / *usage = this process's local (VRAM) budget and current usage, bytes.
extern "C" int gpu_wddm_budget(uint64_t* budget, uint64_t* usage) {
    if (!init()) return -1;
    MemoryBudgetQuery q{0, 0};
    MemoryBudget b{};
    auto av = (AdapterVtbl*)((ComObj*)g_adapter)->vtbl;
    const uint32_t kAdapterMemoryBudget = 1;
    if (FAILED(av->query_state(g_adapter, kAdapterMemoryBudget, sizeof(q), &q, sizeof(b), &b))) return -2;
    *budget = b.budget;
    *usage = b.current_usage;
    return 0;
}
