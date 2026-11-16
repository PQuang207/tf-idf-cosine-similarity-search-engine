#include "../include/document.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>


using namespace std;

namespace fs = filesystem;

vector<RawFile> readAllTxtFiles(const std::string& folderPath)
{
    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) //Kiem tra ton tai duong dan va co phai thu muc khong
    {
        throw std::runtime_error("Loi: Thu muc '" + folderPath + "' khong ton tai!");
    }
    vector<fs::path> filePaths;
    for (const auto& entry : fs::directory_iterator(folderPath))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") //Kiem tra co phai file va la file co duoi .txt
        {
            filePaths.push_back(entry.path());
        }
    }

    sort(filePaths.begin(), filePaths.end(),
    [](const fs::path& a, const fs::path&b)
    {
        return a.filename().string() < b.filename().string();
    });

    vector<RawFile> rawFiles;
    for (const auto& path: filePaths)
    {
        ifstream inFile(path);
        if (!inFile.is_open())
        {
            cerr << "[Canh bao] Khong mo duoc file: " << path.filename().string() << " -> Bo qua.\n";
            continue;
        }
        ostringstream buffer;
        buffer << inFile.rdbuf();
        rawFiles.push_back({path.filename().string(), buffer.str()});
    }
    return rawFiles;
}

void Chao()
{
    cout << "Hello world! 123";
}