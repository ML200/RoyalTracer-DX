#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mc {

struct IResourceProvider {
    virtual ~IResourceProvider() = default;
    // Resolves model and texture paths across the resource stack.
    virtual bool exists(const std::string& path) = 0;
    virtual bool read(const std::string& path, std::vector<uint8_t>& out) = 0;
};

class ZipArchive : public IResourceProvider {
public:
    // Indexes a ZIP central directory for on-demand reads.
    bool open(const std::string& path, std::string* err = nullptr);
    bool open_memory(std::vector<uint8_t> bytes, std::string* err = nullptr);

    bool exists(const std::string& name) override { return m_entries.count(name) != 0; }
    bool read(const std::string& name, std::vector<uint8_t>& out) override;
    void list(const std::string& prefix, std::vector<std::string>& out) const;
    size_t entry_count() const { return m_entries.size(); }
    const std::string& path() const { return m_path; }

private:
    struct Entry { uint32_t localOffset; uint32_t compSize; uint32_t uncompSize; uint16_t method; };
    bool index(std::string* err);
    std::string m_path;
    std::vector<uint8_t> m_data;
    std::unordered_map<std::string, Entry> m_entries;
};

class ResourceStack : public IResourceProvider {
public:
    void push(IResourceProvider* p) { m_layers.push_back(p); }
    bool exists(const std::string& path) override {
        for (IResourceProvider* p : m_layers) if (p->exists(path)) return true;
        return false;
    }
    bool read(const std::string& path, std::vector<uint8_t>& out) override {
        for (IResourceProvider* p : m_layers) if (p->exists(path)) return p->read(path, out);
        return false;
    }
private:
    std::vector<IResourceProvider*> m_layers;
};

class MemoryResources : public IResourceProvider {
public:
    void add(const std::string& path, const std::string& text) { m_files[path] = text; }
    bool exists(const std::string& path) override { return m_files.count(path) != 0; }
    bool read(const std::string& path, std::vector<uint8_t>& out) override {
        const auto it = m_files.find(path);
        if (it == m_files.end()) return false;
        out.assign(it->second.begin(), it->second.end());
        return true;
    }
private:
    std::unordered_map<std::string, std::string> m_files;
};

std::vector<uint8_t> make_stored_zip(const std::vector<std::pair<std::string, std::string>>& files);

}
