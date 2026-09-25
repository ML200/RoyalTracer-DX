#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "../rdn/ocean/OceanCommon.h"
#include "../rdn/ocean/OceanQuadtree.h"
#include <complex>
#include <random>
#include <limits>
#include <map>
#include <tuple>
using namespace ocean;
constexpr double pi = 3.14159265358979323846;
void Require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
double Integrate(const Spectrum& s) {
    double total=0, step=std::log(2000.0)/20000;
    for(int i=0;i<20000;++i){double w=s.omegaP*.05*std::exp((i+.5)*step);total+=s.S(w)*w*step;}
    return total;
}
int main() { try {
    Params p; ValidateParams(p);
    // Detail gain cannot modify long waves, must reach its requested value at 2 m,
    // and must stay bounded across the full wavenumber range.
    Require(ShortWaveAmplitude(p, 2*pi/20) == 1.0, "Long-wave detail isolation");
    Require(ShortWaveAmplitude(p, 2*pi/2) == p.shortWaveAmplitude, "Short-wave amplitude endpoint");
    for (int i=0;i<1000;++i) {
        const double gain=ShortWaveAmplitude(p, i*.5);
        Require(gain>=1.0 && gain<=p.shortWaveAmplitude, "Bounded detail gain");
    }
    Params detailOff=p; detailOff.shortWaveAmplitude=1;
    Require(PredictElevationVariance(p)>PredictElevationVariance(detailOff), "Height prediction includes detail");
    Require(!p.legacySubsurface, "Open ocean must not use solid SSS by default");
    p.significantHeight=2.5f; p.peakPeriod=7;
    Spectrum s; s.Init(p);
    Require(std::abs(4*std::sqrt(Integrate(s))-2.5)<1e-5,"Explicit Hm0 normalization");
    Require(std::abs(s.omegaP-2*pi/7)<1e-12,"Peak period");
    double maxDirectionError=0;
    for(double r : {.4,.8,1.,1.5,2.,5.,10.}) {
        double integral=0;
        for(int j=0;j<8192;++j)integral+=s.D(s.omegaP*r,-pi+(j+.5)*2*pi/8192)*2*pi/8192;
        maxDirectionError=std::max(maxDirectionError,std::abs(integral-1));
    }
    std::cout<<"Directional integral maximum error "<<maxDirectionError<<'\n';
    Require(maxDirectionError<.003,"Directional density normalization");
    double maxPartitionError=0;
    for(int i=0;i<10000;++i) {
        double k=CascadeFundamental(0)*1.001*std::pow(CascadeNyquist(3)/CascadeFundamental(0)/1.002,i/9999.0);
        double weight=0;for(int c=0;c<4;++c)weight+=CascadeWeight(c,k);
        maxPartitionError=std::max(maxPartitionError,std::abs(weight-1));
    }
    Require(maxPartitionError<1e-12,"Cascade power partition");
    // Swell energy and bearing are independent of wind. Integrate actual production Cartesian PSD.
    p.swellHeight=1.7f;p.swellDirectionDeg=73;p.swellSpreadDeg=12;
    double swellEnergy=0,sx=0,sz=0;
    const double dklog=std::log(100.0)/1024;
    const double kp=std::pow(2*pi/p.swellPeriod,2)/kGravity;
    for(int i=0;i<1024;++i) {
        double k=kp*.1*std::exp((i+.5)*dklog);
        for(int j=0;j<1024;++j) {
            double a=(j+.5)*2*pi/1024,x=k*std::sin(a),z=k*std::cos(a);
            double e=SwellDensity(p,x,z)*k*k*dklog*2*pi/1024;
            swellEnergy+=e;sx+=e*std::sin(a);sz+=e*std::cos(a);
        }
    }
    Require(std::abs(4*std::sqrt(swellEnergy)-p.swellHeight)<1e-5,"Swell Hm0 normalization");
    Require(std::abs(std::atan2(sx,sz)*180/pi-73)<1e-6,"Swell bearing");
    p.significantHeight=0;p.swellHeight=0;s.Init(p);
    Require(Integrate(s)==0 && PredictElevationVariance(p)==0,"Zero energy");
    p.windSpeed=std::numeric_limits<float>::quiet_NaN();bool rejected=false;
    try{ValidateParams(p);}catch(const std::invalid_argument&){rejected=true;}
    Require(rejected,"NaN parameter rejected");
    // Ensemble normalization uses asymmetric direction PSD and the declared Hermitian construction.
    std::mt19937 rng(7);std::normal_distribution<double> normal;
    constexpr int count=64, trials=4000;
    std::array<double,count> power{};
    double target=0;for(int i=1;i<count;++i){if(i==32)continue;power[i]=.2+std::sin(i*.2)*.1;target+=power[i];}
    double sum=0,sum2=0;
    for(int t=0;t<trials;++t){
        std::array<std::complex<double>,count> h0{};
        for(int i=0;i<count;++i)h0[i]=.5*std::sqrt(power[i])*std::complex<double>(normal(rng),normal(rng));
        double energy=0;
        for(int i=0;i<count;++i){auto phase=std::polar(1.,-.137*std::min(i,count-i));auto h=h0[i]*phase+std::conj(h0[(count-i)%count])*std::conj(phase);energy+=std::norm(h);}
        sum+=energy;sum2+=energy*energy;
    }
    double mean=sum/trials,se=std::sqrt((sum2/trials-mean*mean)/trials);
    std::cout<<"Ensemble energy ratio "<<mean/target<<", relative standard error "<<se/target<<'\n';
    Require(std::abs(mean-target)<std::max(.02*target,3*se),"Hermitian ensemble normalization");
    // Raw-moment mixture and covariance basis transforms, including correlated slopes.
    const double x0=.1,z0=.4,x1=-.3,z1=.2,w=.35;
    double mx=w*x0+(1-w)*x1,mz=w*z0+(1-w)*z1;
    double cxx=w*x0*x0+(1-w)*x1*x1-mx*mx;
    double cxz=w*x0*z0+(1-w)*x1*z1-mx*mz;
    double czz=w*z0*z0+(1-w)*z1*z1-mz*mz;
    Require(std::abs(cxx-w*(1-w)*std::pow(x0-x1,2))<1e-15,"Moment covariance");
    Require(cxx*czz-cxz*cxz > -1e-15,"Covariance PSD");
    // Gaussian/GGX radial medians coincide at alpha=sqrt(2 ln2 sigma²); no finite GGX variance assumed.
    double sigma2=.03,alpha2=2*std::log(2.)*sigma2;
    Require(std::abs(1-std::exp(-alpha2/(2*sigma2))-.5)<1e-15,"GGX median fit");
    // Complete composite-map analytic derivatives against central finite differences.
    auto surface=[](double u,double v){std::array<double,3>P{u,0,v};
        for(int j=0;j<3;++j){double a=.13*(j+1),kx=.2+.3*j,kz=.4-.12*j,k=std::hypot(kx,kz),ph=kx*u+kz*v;
            P[0]-=.4*a*kx/k*std::sin(ph);P[1]+=a*std::cos(ph);P[2]-=.4*a*kz/k*std::sin(ph);}return P;};
    double maxDerivative=0;
    for(double eps:{1e-3,1e-4,1e-5}) for(int i=0;i<40;++i){double u=i*.37,v=-i*.18;
        std::array<double,3> du{1,0,0},dv{0,0,1};
        for(int j=0;j<3;++j){double a=.13*(j+1),kx=.2+.3*j,kz=.4-.12*j,k=std::hypot(kx,kz),ph=kx*u+kz*v;
            du[0]-=.4*a*kx*kx/k*std::cos(ph);du[1]-=a*kx*std::sin(ph);du[2]-=.4*a*kx*kz/k*std::cos(ph);
            dv[0]-=.4*a*kx*kz/k*std::cos(ph);dv[1]-=a*kz*std::sin(ph);dv[2]-=.4*a*kz*kz/k*std::cos(ph);}
        auto up=surface(u+eps,v),um=surface(u-eps,v),vp=surface(u,v+eps),vm=surface(u,v-eps);
        for(int j=0;j<3;++j){maxDerivative=std::max(maxDerivative,std::abs((up[j]-um[j])/(2*eps)-du[j]));maxDerivative=std::max(maxDerivative,std::abs((vp[j]-vm[j])/(2*eps)-dv[j]));}
        double det=du[0]*dv[2]-dv[0]*du[2];double gx=(dv[2]*du[1]-du[2]*dv[1])/det,gz=(du[0]*dv[1]-dv[0]*du[1])/det;
        Require(std::abs(gx*du[0]+gz*du[2]-du[1])<1e-12,"Composite chain rule u");
        Require(std::abs(gx*dv[0]+gz*dv[2]-dv[1])<1e-12,"Composite chain rule v");
    }
    Require(maxDerivative<1e-6,"Composite derivatives finite difference");
    std::cout<<"Maximum derivative absolute error "<<maxDerivative<<'\n';
    // Rebase wraps preserve texture coordinates modulo one, including negative origins.
    for(double origin:{-1e7,-1234.,0.,1000.,1e7})for(double L:CascadeLengths()){
        double world=123.456,relative=world-origin,wrap=std::fmod(origin,L);
        Require(std::abs(std::remainder((relative+wrap-world)/L,1.))<1e-9,"Rebase phase");
    }
    // Over an open-ocean fetch the sea is fully developed: its peak sits at the Pierson-Moskowitz
    // omega_p U / g ~ 0.855 with the broad unenhanced shape, so the wave period follows the wind.
    // A short fetch keeps it young, shorter and sharply peaked.
    {double previous=0;
        for(float wind:{3.f,5.f,8.f,11.f}){Params w;w.fetch=250000;w.windSpeed=wind;Spectrum sp;sp.Init(w);
            Require(std::abs(sp.omegaP*wind/kGravity-.855)<.01,"Fully developed peak");
            Require(sp.gamma<1.3,"Fully developed sea is broadly peaked");
            const double period=2*pi/sp.omegaP;Require(period>previous*1.2,"Period grows with the wind");previous=period;}
        Params young;young.windSpeed=20;young.fetch=20000;Spectrum sp;sp.Init(young);
        Require(sp.omegaP*20/kGravity>1.5 && sp.gamma>2.5,"Short fetch keeps the sea young");}
    // The waves shorter than the peak follow the wind: the slope the cascades resolve keeps the share
    // of Cox & Munk's it has at the reference wind, which keeps its spectrum as it was; a gale gets
    // steeper short waves, a breeze calmer ones, and an explicit height is still met.
    {auto coxMunk=[](double U){double a,c;CoxMunkSlopeVariance(U,a,c);return a+c;};
        Params ref;ref.fetch=250000;Spectrum rs;rs.Init(ref);
        Require(rs.equilibriumGain==1.0,"Reference sea keeps its spectrum");
        double all,ramped;ResolvedSlopeMoments(ref,rs,all,ramped);const double share=all/coxMunk(ref.windSpeed);
        for(float wind:{5.f,8.f,15.f,20.f,28.f}){Params w=ref;w.windSpeed=wind;Spectrum sp;sp.Init(w);
            ResolvedSlopeMoments(w,sp,all,ramped);
            Require(std::abs(all/coxMunk(wind)/share-1)<1e-9,"Resolved slope follows Cox-Munk");
            Require(wind>11.f?sp.equilibriumGain>1.0:sp.equilibriumGain<1.0,"Short waves steepen with the wind");}
        Params gale=ref;gale.windSpeed=20;gale.significantHeight=4;gale.peakPeriod=9;Spectrum gs;gs.Init(gale);
        Require(gs.equilibriumGain>1.0 && std::abs(4*std::sqrt(Integrate(gs))-4)<1e-5,"Explicit height kept in a gale");}
    // Turbulence gives the short waves their extra chop, up to its cap; the dominant waves and
    // centimetre ripples keep the physical displacement.
    {Params c;Spectrum sp;sp.Init(c);const ChopBand band=ShortChopBand(sp.omegaP);
        const double kp=sp.omegaP*sp.omegaP/kGravity,extra=ShortWaveChop(c);
        Require(std::abs(ChopGainAt(kp,band,extra)-1)<1e-12,"Dominant waves keep the physical displacement");
        Require(std::abs(ChopGainAt(10*kp,band,extra)-1-extra)<1e-12,"Short waves carry the full extra chop");
        Require(std::abs(ChopGainAt(100,band,extra)-1)<1e-12,"Centimetre ripples keep the physical displacement");
        double previous=-1;
        for(float t:{0.f,.5f,1.f,2.f,3.f}){c.turbulence=t;const double e=ShortWaveChop(c);
            Require(e>=previous && e<=kMaxShortChop,"Short-wave chop grows with turbulence up to its cap");previous=e;}
        c.turbulence=1;Params gale=c;gale.windSpeed=20;Params breeze=c;breeze.windSpeed=5;
        Require(ShortWaveChop(gale)>ShortWaveChop(c) && ShortWaveChop(breeze)<ShortWaveChop(c),"Wind forcing sharpens the short waves");
        c.turbulence=0;Require(ShortWaveChop(c)==0,"Untroubled sea keeps every crest rounded");
        Params off;off.foamCoverage=1;Require(BreakingThreshold(off)==kBreakingJacobian,"Physical breaking point by default");}
    // Whitecaps cover a share of what is measured at sea for the wind, whatever the turbulence, and
    // never more than the cap however hard it blows or the control is set.
    {Params p;Require(std::abs(WhitecapCover(p)-kWhitecapShown*WhitecapCoverage(p.windSpeed))<1e-12,"Calibrated sea follows the measured cover");
        Params storm=p;storm.windSpeed=30;storm.foamCoverage=2;Require(WhitecapCover(storm)==kWhitecapMax,"The cover is capped");
        Params wild=p;wild.turbulence=3;Params steady=p;steady.turbulence=0;
        Require(WhitecapCover(wild)==WhitecapCover(p) && WhitecapCover(steady)==WhitecapCover(p),"Turbulence leaves the whitecaps alone");
        Params off=p;off.foamCoverage=0;Require(WhitecapCover(off)==0,"Switched-off foam carries none");
        Params gale=p;gale.windSpeed=20;Params breeze=p;breeze.windSpeed=6;
        Require(WhitecapCover(gale)>5*WhitecapCover(p) && WhitecapCover(breeze)*5<WhitecapCover(p),"Cover rises steeply with the wind");}
    // Turbulence scatters the wind sea off its heading: more of it crosswind, the same total.
    {Params calm,wild;calm.turbulence=.5f;wild.turbulence=2.5f;Spectrum a,b;a.Init(calm);b.Init(wild);
        const double w=a.omegaP*1.2;
        Require(b.D(w,pi/3)>2*a.D(w,pi/3) && b.D(w,0)<a.D(w,0),"Turbulence widens the directional spread");}
    // Solid foam is an opaque raft of the foam albedo; thin foam lets the water through. A bubble
    // cloud under clear water returns turquoise - red lost to the water first - and a denser one
    // catches more and returns it paler.
    {XMFLOAT4 water{1,1,1,0},kd;XMFLOAT3 tf=WaterAbsorptionRGB(.05f);float sss;uint32_t on;
        ocean::WriteMaterialSlot(OCEAN_FOAM_STEPS-1,water,tf,1.f,1u,.6f,kd,sss,on);
        Require(kd.x==.6f && kd.y==.6f && kd.w==1 && on==0,"Solid foam is the foam albedo");
        ocean::WriteMaterialSlot(OCEAN_MATERIAL_LEVELS-1,water,tf,1.f,1u,.6f,kd,sss,on);
        Require(kd.x==.6f && kd.w==1,"Solid foam hides the cloud under it");
        ocean::WriteMaterialSlot(1,water,tf,1.f,1u,.6f,kd,sss,on);
        Require(kd.x==.6f && kd.w<.1f && on==1,"Thin foam is translucent");
        XMFLOAT4 thin,dense;
        ocean::WriteMaterialSlot(OCEAN_FOAM_STEPS,water,tf,1.f,1u,.6f,thin,sss,on);
        ocean::WriteMaterialSlot(OCEAN_FOAM_STEPS*(OCEAN_BUBBLE_STEPS-1),water,tf,1.f,1u,.6f,dense,sss,on);
        Require(thin.x<thin.y && thin.y<thin.z && dense.x<dense.z,"Bubbles under water return turquoise");
        Require(dense.w>thin.w && dense.x/dense.z>thin.x/thin.z && dense.w<1 && on==1,
                "A denser cloud catches more and is paler");}
    // Production quadtree must cover the WHOLE finite square with no slot starvation on jumps, and
    // every stitched edge must name the level of the neighbour that is really there.
    Params geometry;Quadtree tree;planet::CameraView camera{};
    for(int frame=0;frame<60;++frame){camera.position_world={double((frame%7)*1700-5100),double(2+frame*11),double((frame%11)*1300-6500)};
        camera.forward={std::sin(frame*.7f),-.05f*(frame%4),std::cos(frame*.7f)};
        tree.Select(geometry,camera,OCEAN_MAX_TILES,nullptr,frame%2?3000.0:0.0);
        double area=0;std::unordered_set<uint32_t> slots;std::map<std::tuple<int,int64_t,int64_t>,bool> byCell;
        for(auto& t:tree.Tiles()){area+=t.size*t.size;Require(slots.insert(t.slot).second,"Unique tile slot");byCell[{t.level,t.ix,t.iy}]=true;}
        Require(tree.DroppedCount()==0,"Tile budget or slot starvation");
        Require(std::abs(area-4.0*geometry.extent*geometry.extent)<1,"Full ocean coverage");
        const int dx[4]={-1,1,0,0},dy[4]={0,0,-1,1};
        for(auto& t:tree.Tiles())for(int e=0;e<4;++e){
            const uint32_t k=(t.stitch>>(OCEAN_STITCH_BITS*e))&OCEAN_STITCH_MASK;
            Require(k<=(uint32_t)Quadtree::kMaxLevelStep,"Level step within balance");
            if(!k)continue;
            const int64_t nx=((int64_t)t.ix+dx[e])>>k,ny=((int64_t)t.iy+dy[e])>>k;
            Require(byCell.count({t.level-(int)k,nx,ny}),"Stitch names the coarser neighbour");
        }
    }
    // The default sea seen from deck height stays far below the four million triangles a full
    // budget of 64^2 tiles used to cost.
    {Params sea;sea.seaLevelY=5;planet::CameraView deck{};deck.position_world={0,33,150};deck.forward={0,-.008f,-1};deck.fov_y=1.047f;deck.aspect=16.f/9;
        Quadtree t;t.Select(sea,deck,OCEAN_MAX_TILES,nullptr,3100.0);
        const size_t triangles=t.Tiles().size()*(size_t)OCEAN_TILE_TRIS;
        std::cout<<"Deck view: "<<t.Tiles().size()<<" tiles, "<<triangles<<" triangles\n";
        Require(triangles<800000,"Deck-view triangle budget");}
    std::cout<<"PASS: production spectra, wind-driven development, normalization, swell, partition, parameter validation, numerical surface references, rebase, crest strain, whitecap cover, spreading, foam ramp, full quadtree coverage, stitching and 60 camera jumps\n";
    return 0;
}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;} }

