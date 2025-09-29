// Windows-specific implementation: isolate Windows headers here so they
// don't clash with std::byte due to "using namespace std" in other TUs.

#include <iostream>
#if defined(_MSC_VER)
// Disable std::byte in MSVC STL to avoid 'byte' ambiguity with Windows headers.
// This must be defined before including any standard headers that might pull
// <cstddef>.
#ifndef _HAS_STD_BYTE
#define _HAS_STD_BYTE 0
#endif
#endif

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#ifndef _WIN32_DCOM
#define _WIN32_DCOM
#endif

#include <chrono>
#include <cstdlib>
#include <optional>
#include <thread>


#include <wbemidl.h>
#include <windows.h>

#pragma comment(lib, "wbemuuid.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")

static inline std::optional<std::string> getenv_safe(const char* key) {
#ifdef _WIN32
    char* buf = nullptr;
    size_t len = 0;
    if (_dupenv_s(&buf, &len, key) == 0 && buf) {
        std::string val(buf, len ? len - 1 : 0); // drop trailing '\0'
        free(buf);
        return val;
    }
    return std::nullopt;
#else
    if (const char* s = std::getenv(key)) {
        return std::string(s);
    }
    return std::nullopt;
#endif
}

static int env_int(const char* key, int defv) {
    if (auto val = getenv_safe(key)) {
        try {
            return std::stoi(*val);
        } catch (const std::invalid_argument&) {
        } catch (const std::out_of_range&)   {
        }
    }
    return defv;
}

static double env_double(const char* key, double defv) {
    if (auto val = getenv_safe(key)) {
        try {
            return std::stod(*val);
        } catch (const std::invalid_argument&) {
        } catch (const std::out_of_range&)   {
        }
    }
    return defv;
}

std::optional<double> read_cpu_temp_c_windows() {
  HRESULT hr;
  hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  bool com_inited = SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE;
  if (!com_inited)
    return std::nullopt;

  CoInitializeSecurity(nullptr, -1, nullptr, nullptr, RPC_C_AUTHN_LEVEL_DEFAULT,
                       RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE,
                       nullptr);

  IWbemLocator *pLoc = nullptr;
  IWbemServices *pSvc = nullptr;
  std::optional<double> result;

  do {
    hr = CoCreateInstance(CLSID_WbemLocator, 0, CLSCTX_INPROC_SERVER,
                          IID_IWbemLocator, (LPVOID *)&pLoc);
    if (FAILED(hr) || !pLoc)
      break;

    hr = pLoc->ConnectServer(BSTR(L"ROOT\\WMI"), nullptr, nullptr, 0, 0, 0, 0,
                             &pSvc);
    if (FAILED(hr) || !pSvc)
      break;

    hr = CoSetProxyBlanket(pSvc, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, nullptr,
                           RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE,
                           nullptr, EOAC_NONE);
    if (FAILED(hr))
      break;

    IEnumWbemClassObject *pEnum = nullptr;
    hr = pSvc->ExecQuery(
        BSTR(L"WQL"),
        BSTR(L"SELECT CurrentTemperature FROM MSAcpi_ThermalZoneTemperature"),
        WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY, nullptr, &pEnum);
    if (FAILED(hr) || !pEnum)
      break;

    double bestC = -1e9;
    for (;;) {
      IWbemClassObject *pObj = nullptr;
      ULONG ret = 0;
      hr = pEnum->Next(2000, 1, &pObj, &ret);
      if (FAILED(hr) || ret == 0)
        break;
      VARIANT vt;
      VariantInit(&vt);
      if (SUCCEEDED(pObj->Get(L"CurrentTemperature", 0, &vt, 0, 0)) &&
          (vt.vt == VT_UINT || vt.vt == VT_I4)) {
        const double kelvin_tenths =
            (vt.vt == VT_UINT) ? vt.uintVal : vt.intVal;
        const double celsius = kelvin_tenths / 10.0 - 273.15;
        if (celsius > bestC)
          bestC = celsius;
      }
      VariantClear(&vt);
      pObj->Release();
    }
    if (pEnum)
      pEnum->Release();
    if (bestC > -1e8)
      result = bestC;

  } while (false);

  if (pSvc)
    pSvc->Release();
  if (pLoc)
    pLoc->Release();
  CoUninitialize();
  return result;
}

void cooldown_between_jobs_windows() {
  const int min_sleep_sec = env_int("UNIF_COOLDOWN_MIN_SEC", 0);
  const int max_sleep_sec = env_int("UNIF_COOLDOWN_MAX_SEC", 240);
  const int poll_sec = env_int("UNIF_COOLDOWN_POLL_SEC", 10);
  const double target_temp_c = env_double("UNIF_COOLDOWN_TARGET_C", 75.0);

  int slept = 0;
  if (min_sleep_sec > 0) {
    std::this_thread::sleep_for(std::chrono::seconds(min_sleep_sec));
    slept += min_sleep_sec;
  }
  for (;;) {
    if (slept >= max_sleep_sec)
      break;
    auto t = read_cpu_temp_c_windows();
    if (!t.has_value()) {
      // std::cout << "No sensor output" << std::endl;
      break;
    }
    if (t.value() <= target_temp_c) {
      // std::cout << "Temperature " << t.value() << " is below threshold " << target_temp_c << std::endl;
      break;
    }
    // std::cout << "Temperature " << t.value() << " is above threshold " << target_temp_c << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(poll_sec));
    slept += poll_sec;
  }
}