#include "../include/tokenizer.h"
#include <fstream>
#include <cctype>
#include <stdexcept>
#include <iostream>

using namespace std;

static bool isAsciiAlnum(char c)
{
    unsigned char uc = static_cast<unsigned char>(c);
    return uc < 128 && isalnum(uc);
}

static bool isAllAlpha(const string& s)
{
    if (s.empty())
        return false;
    for (char ch: s)
        if (!isalpha(static_cast<unsigned char>(ch)))
            return false;
    return true;
}

unordered_set<string> loadStopWords(const string& filepath)
{
    ifstream inFile(filepath);
    if (!inFile.is_open())
    {
        throw runtime_error("Loi cau hinh: Khong tim thay hoac khong mo duoc file " + filepath + "!");
    }
    unordered_set<string> stopWords;
    string line;
    while (getline(inFile, line))
    {
        vector<string> wordsInLine = tokenize(line);
        for (const string& word : wordsInLine)
            stopWords.insert(word);
    }
    return stopWords;
}

vector<string> tokenize(const string& text)
{
    vector<string> tokens;
    string currentToken;
    size_t n = text.size();
    for (size_t i = 0; i < n; i++)
    {
        char c = text[i];
        if (isAsciiAlnum(c))
        {
            unsigned char uc= static_cast<unsigned char>(c);
            currentToken += static_cast<char>(tolower(uc));
        }
        else
        {
            if (isAllAlpha(currentToken))
            {
                if (i + 1 < n && text[i] == '+' && text[i + 1] == '+' &&
                    (i + 2 == n || !isAsciiAlnum(text[i + 2])))
                    {
                        currentToken += "++";
                        i += 1;
                    }
                else if (text[i] == '#' && (i + 1 == n || !isAsciiAlnum(text[i+1])))
                    currentToken += '#';
            }
            if (!currentToken.empty())
            {
                tokens.push_back(currentToken);
                currentToken.clear();
            }
        }
    }
    if (!currentToken.empty())
    {
        tokens.push_back(currentToken);
        currentToken.clear();
    }
    return tokens;
}

vector<string> preprocess(const string& text, const unordered_set<string>& stopWords)
{
    vector<string> rawTokens = tokenize(text);
    vector<string> cleanTokens;
    for (const string& token : rawTokens)
        if (stopWords.find(token) == stopWords.end())
            cleanTokens.push_back(token);
    return cleanTokens;
}