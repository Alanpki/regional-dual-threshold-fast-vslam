#include "PerformanceLogger.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <string.h>

#include <memory>
#include <thread>
#include <mutex>
#include <atomic>

#include <chrono>

std::atomic<bool> PerformanceLogger::keepRunning(true);
std::string PerformanceLogger::latestTegraStats = "";
std::mutex PerformanceLogger::tegraMutex;

void PerformanceLogger::execTegrastats() {
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("tegrastats --interval 1", "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    while(keepRunning) {
        char buffer[1024];
        std::string result;

        if (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
            result = buffer;
        }
        else {
            std::cerr << "Failed to read tegrastats output" << std::endl;
            break;
        }

        {
            std::lock_guard<std::mutex> lock(tegraMutex);
            latestTegraStats = buffer;
        }
    }
}

void PerformanceLogger::logTegrastats(const std::string& module_name, const std::string& id_type, long unsigned int id, const std::string& phase) {
    const char* home = std::getenv("HOME");
    std::string log_path = std::string(home) + "/JetsonSLAM_tegrastats_log.txt";
    std::ofstream log_file(log_path, std::ios::app);
    std::string tegra_output = getTegrastats();

    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    long long timestamp = duration.count();

    while(tegra_output.empty()){
        tegra_output = getTegrastats();
    }

    if (log_file.is_open()) {
        log_file << "Timestamp: " << timestamp << " | "
                 << "Module: " << module_name << " | "
                 << id_type << ": " << id << " | "
                 << "Phase: " << phase << " | "
                 << tegra_output;
    }
    else {
        std::cout << "Error: Could not open log file." << std::endl;
    }

    std::cout << module_name << " " << tegra_output << std::endl;
}

std::string PerformanceLogger::getTegrastats() {
    std::lock_guard<std::mutex> lock(tegraMutex);

    return latestTegraStats;
}