#include "../../rdn/stdafx.h"
#include "Ocean.h"

void Ocean::Init(const Params& params, Renderer& renderer) {
    ocean::ValidateParams(params);
    m_params = params;
    m_params.enabled = true;
    renderer.GetOcean().Configure(m_params);
    LOG(L"[Ocean] enabled: wind " << m_params.windSpeed << L" m/s from " << m_params.windDirectionDeg << L" deg, fetch "
                                  << (m_params.fetch / 1000.0f) << L" km, extent "
                                  << (m_params.extent / 1000.0f) << L" km");
}

void Ocean::SetParams(const Params& params, Renderer& renderer) {
    ocean::ValidateParams(params);
    m_params = params;
    renderer.GetOcean().Configure(m_params);
}

