// Included by the standalone GPU runner after its upload/readback helpers.
#include <thread>
void VerifyLightPacking(Runner& runner) {
    constexpr uint32_t count = 8193u, leafBase = 1000000u, headerOffset = 7u;
    uint32_t random = 17u;
    auto unit = [&]() {
        random = 1664525u * random + 1013904223u;
        return float(random >> 8u) * (1.f / 16777216.f);
    };
    // Root spheres of very different scales and offsets: ordinary, degenerate, tiny, far away,
    // huge coordinates, and extreme extents.
    struct Domain {
        XMFLOAT3 center;
        float extent;
    };
    const Domain cases[] = {{{0, 0, 0}, 16.f},
                            {{0, 0, 0}, 0.f},
                            {{0, 0, 0}, .001f},
                            {{1000000, -2000000, 0}, 1.f},
                            {{100000000, 100000000, -100000000}, 256.f},
                            {{-1e30f, -1e20f, 1e-20f}, 1e20f}};
    std::vector<lt::LightBLASNodeGpu> nodes(count);
    for (const auto& domain : cases) {
        for (uint32_t i = 0; i < count; ++i) {
            auto& n = nodes[i];
            n = {};
            // Every member sphere lies inside the root sphere, as the builder guarantees.
            const XMFLOAT3 dir = lt::normalize3({unit() * 2 - 1, unit() * 2 - 1, unit() * 2 - 1});
            const float offset = unit() * .9f * domain.extent;
            n.mean = lt::add3(domain.center, lt::mul3(dir, offset));
            n.radius = (i % 7u ? unit() : 1.f) * (domain.extent - offset);
            const float sigma = (i % 3u ? unit() : 1.f) * n.radius;
            n.variance = sigma * sigma;
            const float length = i % 11u ? unit() * .9f : 0.f;
            n.rbar = lt::mul3(lt::normalize3({unit() * 2 - 1, unit() * 2 - 1, unit() * 2 - 1}), length);
            n.cosTheta_o = i % 5u ? unit() * 2 - 1 : 1.f;
            n.power = std::ldexp(1.f + unit(), int(i % 101u) - 50);
            n.triFirst = i;
            n.triCount = 1;
        }
        nodes[0].mean = domain.center;
        nodes[0].radius = domain.extent;
        nodes[0].variance = domain.extent * domain.extent / 9.f;
        nodes[0].firstChild = 1;
        nodes[0].childCount = 4;
        nodes[0].triCount = 0;
        auto packed = lt::PackBLAS(nodes, leafBase);
        Require(packed.size() == nodes.size() + 1u, "Packed BLAS must have exactly one mesh header");
        const float meshUnit = lt::LightBLASUnit(domain.extent);
        for (uint32_t i = 0; i < count; ++i) {
            const auto decoded = lt::UnpackBLASNode<lt::LightBLASNodeGpu>(packed.data(), i);
            const auto& original = nodes[i];
            Require(std::memcmp(&decoded.mean, &original.mean, sizeof(XMFLOAT3)) == 0, "Packed light mean lost FP32 precision");
            Require(decoded.radius >= original.radius && decoded.radius <= original.radius * 1.002f + 4e-7f * meshUnit,
                    "CPU decoded light radius shrank or widened beyond the half rounding");
            Require(decoded.variance >= original.variance * 0.999999f &&
                        std::sqrt(decoded.variance) <= std::sqrt(original.variance) * 1.002f + 4e-7f * meshUnit,
                    "CPU decoded light variance shrank or widened beyond the half rounding");
            Require(decoded.power == original.power, "Light power lost FP32 precision");
            Require(decoded.cosTheta_o <= original.cosTheta_o, "CPU decoded light cone narrowed");
            const float lengthIn = lt::length3(original.rbar), lengthOut = lt::length3(decoded.rbar);
            Require(lengthOut <= lengthIn * 1.00001f + 1e-7f && lengthOut >= lengthIn * .999f - 1e-3f,
                    "Mean resultant length sharpened or lost too much precision");
            if (lengthIn > 1e-3f)
                Require(lt::dot3(lt::normalize3(decoded.rbar), lt::normalize3(original.rbar)) > std::cos(2e-3f),
                        "Mean resultant axis lost the octahedral precision");
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
                Require(r.x == 1, "GPU decoded light extent shrank or lost the mean");
                Require(r.y == 1, "GPU light power changed");
                Require(r.z == 1, "GPU packed topology/rebased leaf index changed");
                Require(r.w == 1, "GPU packed cone or emission lobe lost its support");
            }
        }
        // The disabled path must preserve full-precision values, with no mesh header.
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
                        "GPU full-precision decoding lost extent, power, topology, cone or lobe");
        }
        runner.compactNodes=true;
    }
    // The streamed builder must also retain only reachable nodes and leaf trails.
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
    auto placeLeaf = [](lt::TLASExtraLeaf& l, float y) {
        l.aabb = {{float(l.slot), y, 0}, {float(l.slot) + 1, y + 1, 1}};
        l.sg.mean = {float(l.slot) + .5f, y + .5f, .5f};
        l.sg.radius = .87f;
        l.sg.variance = .25f;
        l.sg.rbar = {0, 0, .5f};
        l.sg.theta_o = 0.f;
        l.sg.power = 1.f;
    };
    for (uint32_t i = 0; i < leaves.size(); ++i) {
        auto& l = leaves[i];
        l.slot = i;
        l.power = 1;
        placeLeaf(l, 0.f);
    }
    lt::IncrementalTLAS tree;
    tree.rebuild(leaves, 32);
    VerifyNoUnreachableNodes(tree.nodes());
    const auto before = tree.nodes();
    const auto trails = tree.trails();
    for (auto& l : leaves)
        placeLeaf(l, 2.f);
    tree.update(leaves, 32);
    Require(lt::SameLightTreeTopology(before, tree.nodes()) && trails == tree.trails(),
            "Refit changed stable node identities");
    Require(tree.nodes()[0].mean.y > before[0].mean.y + 1.5f && tree.nodes()[0].power == before[0].power,
            "Refit did not move the root cluster with its leaves");
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
    std::cout << "Packed light nodes: CPU/GPU conservative extents, lobes and cones, FP32 means and power, stream "
                 "offsets, compact topology and stable refits passed\n";
}
