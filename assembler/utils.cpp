#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>

#include "assembler/feature.h"
#include "assembler/file_reader.h"
#include "assembler/utils.h"
#include "deps/murmurhash3/MurmurHash3.h"

namespace assembler {
namespace utils {

std::vector<StringPiece> Split(StringPiece text, const std::string& delim) {
  std::vector<StringPiece> result;
  size_t token_start = 0;
  if (!text.empty()) {
    for (size_t i = 0; i < text.size() + 1; i++) {
      if ((i == text.size()) ||
          (delim.find_first_of(text[i]) != std::string::npos)) {
        StringPiece token(text.data() + token_start, i - token_start);
        result.push_back(token);
        token_start = i + 1;
      }
    }
  } else {
    result.push_back(StringPiece(text.data(), 0));
  }
  return result;
}

void PrintExample(const Example& example) {
  std::cout << example.label << ", " << example.weight << ": ";
  for (auto v : example.feature) {
    std::cout.setf(std::ios::left);
    std::cout.width(4);
    std::cout << v << " ";
  }
  std::cout << std::endl;
}

void WriteString(std::ostringstream& oss, const std::string& s) {
  oss.write(s.data(), s.size() * sizeof(char));
  oss.put(0);
}

void WriteInt(std::ostringstream& oss, int v) {
  oss.write(reinterpret_cast<char*>(&v), sizeof(int));
}
void WriteSize(std::ostringstream& oss, size_t v) {
  oss.write(reinterpret_cast<char*>(&v), sizeof(size_t));
}

void WriteFloat(std::ostringstream& oss, float v) {
  oss.write(reinterpret_cast<char*>(&v), sizeof(float));
}

void ReadString(std::istringstream& iss, std::string* s) {
  char c;
  while ((c = iss.get()) != 0) {
    s->push_back(c);
  }
}

void ReadInt(std::istringstream& iss, int* v) {
  iss.read(reinterpret_cast<char*>(v), sizeof(int));
}
void ReadSize(std::istringstream& iss, size_t* v) {
  iss.read(reinterpret_cast<char*>(v), sizeof(size_t));
}

void ReadFloat(std::istringstream& iss, float* v) {
  iss.read(reinterpret_cast<char*>(v), sizeof(float));
}

std::map<std::string, int> ParseMeta(const std::string& meta_str) {
  std::map<std::string, int> meta;
  std::vector<StringPiece> fields = utils::Split(meta_str, " \t");
  for (size_t i = 0; i < fields.size(); ++i) {
    meta[fields[i].ToString()] = i;
  }
  return meta;
}

bool LoadFromDictFile(::tensorflow::Env* env, const std::string& dict_file,
                      int min_count, int top_k,
                      std::unordered_map<std::string, int>* indexer,
                      std::vector<std::string>* keys) {
  indexer->clear();
  FileReader file_reader(env, dict_file);
  if (!file_reader.Init()) {
    std::cerr << "Open dict file " << dict_file << " failed." << std::endl;
    return false;
  }


  std::string line;
  int lineindex = 0;
  while (file_reader.ReadLine(&line)) {
    if (top_k > 0 && static_cast<int>(indexer->size()) >= top_k) {
      break;
    }

    ++lineindex;
    if (line.empty()) {
      continue;
    }
    auto tokens = utils::Split(line, "|");
    int idx = static_cast<int>(indexer->size());
    std::string key = tokens[0].ToString();
    if (tokens.size() == 2) {
      // key freq
      int freq = -1;
      try {
        freq = std::stoi(tokens[1].ToString());
      } catch (const std::exception& e) {
        std::cerr << "dict file " << dict_file << "format error[1] in line "
                  << lineindex << "." << std::endl;
        return false;
      }
      if (freq < min_count) {
        continue;
      }
      if (indexer->count(key) > 0) {
        std::cerr << "Duplicated key '" << key << "' in dict '" << dict_file
                  << "'." << std::endl;
        return false;
      }
    } else {
      std::cerr << "dict file " << dict_file << "format error[2] in line "
                << lineindex << "." << std::endl;
      return false;
    }
    (*indexer)[key] = idx;
    keys->push_back(key);
  }
  std::cout << "dict file = " << dict_file << ", size = " << keys->size()
            << ", min_count = " << min_count << ", top_k = " << top_k
            << std::endl;
  for (size_t i = 0; i < std::min(5UL, keys->size()); i++) {
    std::cout << "    " << (*keys)[i] << std::endl;
  }

  return true;
}

void TotalSplit(const StringPiece& input,
                std::vector<std::vector<StringPiece>>* pieces) {
  auto tokens = Split(input, "$");
  for (const auto& token : tokens) {
    pieces->push_back(Split(token, "\t"));
  }
}

uint32_t MurMurHash3(const std::string& key) {
  uint32_t out;
  MurmurHash3_x86_32(key.data(), static_cast<int>(key.size()), 1234567, &out);
  return out;
}

}  // namespace utils
}  // namespace assembler
