// Included by the standalone GPU runner after its upload/readback helpers.
#include <thread>
void VerifyLightPacking(Runner& runner) {
    constexpr uint32_t count = 8193u, leafBase = 1000000u, headerOffset = 7u;
    uint32_t random = 17u;
    auto unit = [&]() {
        random = 1664525u * random + 1013904223u;
        return float(random >> 8u) * (1.f / 16777216.f);
    };
    const lt::Aabb cases[] = {{{-16, -8, -2}, {16, 8, 2}},
                              {{0, 0, 0}, {0, 0, 0}},
                              {{0, 0, 0}, {.001f, .002f, 0}},
                              {{1000000, -2000000, 0}, {1000001, -1999999, .01f}},
                              {{100000000, 100000000, -100000000}, {100000256, 100000512, -99999488}},
                              {{-1e30f, -1e20f, 1e-20f}, {1e30f, 1e20f, 2e-20f}}};
    std::vector<lt::LightBLASNodeGpu> nodes(count);
    for (const auto& box : cases) {
        for (uint32_t i = 0; i < count; ++i) {
            auto& n = nodes[i];
            n = {};
            const float lo[3] = {box.mn.x, box.mn.y, box.mn.z}, hi[3] = {box.mx.x, box.mx.y, box.mx.z};
            float lower[3], upper[3];
            for (uint32_t a = 0; a < 3; ++a) {
                const float t = unit(), u = i % 3u ? unit() : t;
                lower[a] = (1.f - (std::min)(t, u)) * lo[a] + (std::min)(t, u) * hi[a];
                upper[a] = (1.f - (std::max)(t, u)) * lo[a] + (std::max)(t, u) * hi[a];
                lower[a] = std::clamp(lower[a], lo[a], hi[a]);
                upper[a] = std::clamp(upper[a], lo[a], hi[a]);
            }
            n.bmin = {lower[0], lower[1], lower[2]};
            n.bmax = {upper[0], upper[1], upper[2]};
            n.axis = lt::normalize3({unit() * 2 - 1, unit() * 2 - 1, unit() * 2 - 1});
            n.cosTheta_o = i % 5u ? unit() * 2 - 1 : 1.f;
            n.sinTheta_o = std::sqrt((std::max)(0.f, 1.f - n.cosTheta_o * n.cosTheta_o));
            n.power = std::ldexp(1.f + unit(), int(i % 101u) - 50);
            n.triFirst = i;
            n.triCount = 1;
        }
        nodes[0].bmin = box.mn;
        nodes[0].bmax = box.mx;
        nodes[0].firstChild = 1;
        nodes[0].childCount = 4;
        nodes[0].triCount = 0;
        auto packed = lt::PackBLAS(nodes, leafBase);
        Require(packed.size() == nodes.size() + 1u, "Packed BLAS must have exactly one bounds header");
        for (uint32_t i = 0; i < count; ++i) {
            const auto decoded = lt::UnpackBLASNode<lt::LightBLASNodeGpu>(packed.data(), i);
            const auto& original = nodes[i];
            Require(decoded.bmin.x <= original.bmin.x && decoded.bmin.y <= original.bmin.y &&
                        decoded.bmin.z <= original.bmin.z && decoded.bmax.x >= original.bmax.x &&
                        decoded.bmax.y >= original.bmax.y && decoded.bmax.z >= original.bmax.z,
                    "CPU decoded light bounds shrank");
            Require(decoded.power == original.power, "Light power lost FP32 precision");
        }
        // Exercise nonzero mesh offsets and streamed leaf-index rebasing.
        packed.insert(packed.begin(), headerOffset, lt::LightBLASNodePacked{});
        auto expected = nodes;
        std::vector<lt::LightTLASNodeGpu> tlas(count);
        for (uint32_t i = 0; i < count; ++i) {
            if (!expected[i].childCount)
                expected[i].triFirst += leafBase;
            std::memcpy(&tlas[i], &expected[i], 48);
            tlas[i].firstChild = expected[i].firstChild;
            tlas[i].childCount = expected[i].childCount;
            tlas[i].slot = expected[i].triFirst;
        }
        auto original = runner.Upload(expected);
        runner.Srv(original.Get(), 20, sizeof(expected[0]));
        auto bg = runner.Upload(packed);
        runner.Srv(bg.Get(), 10, sizeof(packed[0]));
        auto tg = runner.Upload(lt::PackTLAS(tlas));
        runner.Srv(tg.Get(), 9, sizeof(lt::LightTLASNodePacked));
        for (uint32_t mode : {47u, 48u}) {
            const auto results = runner.LearningSamples(count, count, mode, headerOffset, 0);
            for (const auto& r : results) {
                Require(r.x == 1, "GPU decoded light bounds shrank");
                Require(r.y == 1, "GPU light power changed");
                Require(r.z == 1, "GPU packed topology/rebased leaf index changed");
                Require(r.w == 1, "GPU packed cone lost angular support");
            }
        }
        // Disabled compaction: full precision, no mesh header.
        auto fullWords = lt::EncodeLightBLAS(nodes, false, leafBase);
        Require(fullWords.size()*sizeof(uint32_t) == expected.size()*sizeof(expected[0]),
                "Disabled compaction allocated a header or used the wrong stride");
        Require(std::memcmp(fullWords.data(), expected.data(), fullWords.size()*sizeof(uint32_t)) == 0,
                "Disabled compaction altered full-precision nodes or leaf rebasing");
        fullWords.insert(fullWords.begin(), headerOffset*16u, 0u);
        auto fullBg=runner.Upload(fullWords);runner.Srv(fullBg.Get(),10,64u);
        auto fullTg=runner.Upload(tlas);runner.Srv(fullTg.Get(),9,64u);
        runner.compactNodes=false;
        for (uint32_t mode : {47u,48u}) {
            const auto results=runner.LearningSamples(count,count,mode,headerOffset,0);
            for (const auto& r:results)
                Require(r.x==1 && r.y==1 && r.z==1 && r.w==1,
                        "GPU full-precision decoding lost bounds, power, topology or cone support");
        }
        runner.compactNodes=true;
    }
    // The streamed builder also keeps only reachable nodes and leaf trails.
    std::vector<LightTriangle> tris(257);
    for (uint32_t i = 0; i < tris.size(); ++i) {
        auto& t = tris[i];
        t.x = {float(i), 0, 0};
        t.y = {float(i), 1, 0};
        t.z = {float(i) + .5f, 0, 0};
        t.weight = 1;
    }
    lt::LightTreeBuilder::SingleBLAS single;
    lt::LightTreeBuilder::BuildSingleBLAS(tris, single);
    VerifyNoUnreachableNodes(single.nodes);
    Walk(single.nodes, 0, 0, 0, 1.f, [&](const auto& n, uint64_t trail, float) {
        Require(single.trails.at(single.leafTriLocal.at(n.triFirst)) == trail,
                "Streamed compaction changed a triangle trail");
    });
    // Normal refits preserve IDs. Packing also keeps tombstone slot sentinels.
    std::vector<lt::TLASExtraLeaf> leaves(16);
    for (uint32_t i = 0; i < leaves.size(); ++i) {
        auto& l = leaves[i];
        l.slot = i;
        l.aabb = {{float(i), 0, 0}, {float(i) + 1, 1, 1}};
        l.power = 1;
    }
    lt::IncrementalTLAS tree;
    tree.rebuild(leaves, 32);
    VerifyNoUnreachableNodes(tree.nodes());
    const auto before = tree.nodes();
    const auto trails = tree.trails();
    for (auto& l : leaves) {
        l.aabb.mn.y += 2;
        l.aabb.mx.y += 2;
    }
    tree.update(leaves, 32);
    Require(lt::SameLightTreeTopology(before, tree.nodes()) && trails == tree.trails(),
            "Refit changed stable node identities");
    leaves.erase(leaves.begin() + 3);
    tree.update(leaves, 32);
    const auto after = DecodeTLAS(lt::PackTLAS(tree.nodes()));
    Require(lt::SameLightTreeTopology(tree.nodes(), after), "Packed incremental tree lost tombstone identity");
    lt::LightTreeRefitManager manager;
    lt::RequestIncrementalRefit(manager, {}, {}, {}, {}, leaves, 32, 1, &tree, false);
    lt::TLASRefitResult result;
    while (!manager.PollResult(result))
        std::this_thread::yield();
    Require(result.packedNodes.size() == result.nodes.size(), "Worker did not publish packed nodes with topology");
    lt::RequestIncrementalRefit(manager, {}, {}, {}, {}, leaves, 32, 2, &tree, false, false);
    while (!manager.PollResult(result))
        std::this_thread::yield();
    Require(!result.nodes.empty() && result.packedNodes.empty(), "Disabled compaction still encodes refit nodes");
    std::cout << "Packed light nodes: CPU/GPU conservative bounds and cones, FP32 power, stream offsets, compact "
                 "topology and stable refits passed\n";
}
