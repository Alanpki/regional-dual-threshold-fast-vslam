// [20261115] 消融實驗開關（重投用）。環境變數 ABLATION=四位元字串，順序：自適應FAST／兩階段匹配／關鍵幀決策／自適應三角化
// 未設定或格式不對 → 全開（"1111"），程式行為和原本完全相同
#pragma once
#include <cstdlib>
#include <string>
#include <iostream>
namespace Jetson_SLAM {
inline bool AblationOn(int k) {
    static const std::string s = []() {
        const char* e = std::getenv("ABLATION");
        std::string v = (e && std::string(e).size() == 4 && std::string(e).find_first_not_of("01") == std::string::npos) ? std::string(e) : std::string("1111");
        std::cout << "[ABLATION] " << v << " (FAST/TwoStage/KeyFrame/Triangulation)" << std::endl;
        return v;
    }();
    return s[k] == '1';
}
}
