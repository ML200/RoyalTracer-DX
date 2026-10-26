// Included by the standalone GPU runner after its upload/readback helpers.
#include <thread>
void VerifyLightNodeEncoding(Runner& runner) {
    constexpr uint32_t count = 8193u, leafBase = 1000000u, headerOffset = 7u;
    uint32_t random = 17u;
    auto unit = [&]() {
        random = 1664525u * random + 1013904223u;
        return float(random >> 8u) * (1.f / 16777216.f);
    };
    std::vector<lt::LightBLASNodeGpu> nodes(count);
    for (uint32_t i = 0; i < count; ++i) {
        auto& n = nodes[i];
        n = {};
        float lower[3], upper[3];
        for (uint32_t a = 0; a < 3; ++a) {
            const float t = unit() * 32 - 16, u = i % 3u ? unit() * 32 - 16 : t;
            lower[a] = (std::min)(t, u);
            upper[a] = (std::max)(t, u);
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
    nodes[0].firstChild = 1;
    nodes[0].childCount = 4;
    nodes[0].triCount = 0;
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
    auto words = lt::EncodeLightBLAS(nodes, leafBase);
    Require(words.size() * sizeof(uint32_t) == expected.size() * sizeof(expected[0]),
            "Encoded light BLAS has a header or the wrong stride");
    Require(std::memcmp(words.data(), expected.data(), words.size() * sizeof(uint32_t)) == 0,
            "Encoding altered light nodes or leaf rebasing");
    // Exercise nonzero mesh offsets, as streamed meshes use.
    words.insert(words.begin(), headerOffset * 16u, 0u);
    auto original = runner.Upload(expected);
    runner.Srv(original.Get(), 20, sizeof(expected[0]));
    auto bg = runner.Upload(words);
    runner.Srv(bg.Get(), 10, 16u);
    auto tg = runner.Upload(tlas);
    runner.Srv(tg.Get(), 9, 16u);
    for (uint32_t mode : {47u, 48u}) {
        const auto results = runner.LearningSamples(count, count, mode, headerOffset, 0);
        for (const auto& r : results)
            Require(r.x == 1 && r.y == 1 && r.z == 1 && r.w == 1,
                    "GPU light node loads lost bounds, power, topology or cone");
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
    // Normal refits preserve IDs.
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
    lt::LightTreeRefitManager manager;
    lt::RequestIncrementalRefit(manager, {}, {}, {}, {}, leaves, 32, 1, &tree, false);
    lt::TLASRefitResult result;
    while (!manager.PollResult(result))
        std::this_thread::yield();
    Require(lt::SameLightTreeTopology(tree.nodes(), result.nodes), "Worker did not publish the refit topology");
    std::cout << "Light nodes: bit-exact encoding and GPU loads, stream offsets and stable refits passed\n";
}
