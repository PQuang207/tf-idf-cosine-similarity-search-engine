#pragma once
#include <string>
#include <vector>
#include <unordered_map>
using namespace std;

struct Document
{
    int id = -1;
    string filename;
    string content;
    unordered_map<string, int> termCount;
    int tokenCount = 0;
    unordered_map<string, double> weights;
    double norm = 0.0;
};

struct RawFile
{
    string filename;
    string content;
};

vector<RawFile> readAllTxtFiles(const string& folderPath);

void Chao();