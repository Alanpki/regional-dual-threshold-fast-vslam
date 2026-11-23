#include <string>
#include <fstream>

#include <atomic>
#include <mutex>


class PerformanceLogger {
public:
    static std::atomic<bool> keepRunning;
    static std::string latestTegraStats;
    static std::mutex tegraMutex;

    // 記錄效能數據
    static void logTegrastats(const std::string& module_name, const std::string& id_type, long unsigned int id, const std::string& phase);
    
    // 執行 tegrastats 並返回結果
    static void execTegrastats();

    static std::string getTegrastats();
};