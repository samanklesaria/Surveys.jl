using Surveys, RCall, DataFrames, DataFramesMeta, Test, CSV, StatsBase

R"""
library(survey)
data(api)

result_and_stderr <- function(result) {
    c(result, SE(result))
}
"""

function r_comp(julia_val, r_val)
    jv = julia_val[1, :total]
    rv = rcopy(rcall(:result_and_stderr, r_val))
    @test all(rv .≈ [jv.sum, sqrt(jv.var)])
end

# Simple Random Sampling

@rget apisrs
r_comp(
    @combine(apisrs, :total = π_sum(:enroll, Int(:fpc[1]))),
    R"svytotal(~enroll, svydesign(id=~1, fpc=~fpc, data=apisrs))")

# Stratified Sampling

@rget apistrat
strat_jl = @chain apistrat begin
    @groupby(:stype)
    @combine(:subtotal = π_sum(:enroll, Int(:fpc[1])))
    @combine(:total = sum(:subtotal))
end
strat_r = R"svytotal(~enroll, svydesign(id=~1, fpc=~fpc, strata=~stype, data=apistrat))"
r_comp(strat_jl, strat_r)

# One Stage Sampling

cal_crime = DataFrame(CSV.File("test/cal_crime.csv"))
gdf = groupby(cal_crime, :county)
N_counties = cal_crime[1, :fpc]
@rput cal_crime
cluster1_jl = @chain gdf begin
    @combine(:subtotal = sum(:Burglary))
    @combine(:total = π_sum(:subtotal, N_counties))
end
cluster1_r = R"svytotal(~Burglary, svydesign(id=~county, fpc=~fpc, data=cal_crime))"
r_comp(cluster1_jl, cluster1_r)

# Two Stage Sampling

stage2 = combine(gdf, g -> g[sample(1:size(g, 1),
    min(size(g, 1), 5); replace=false), :])
cs = @combine(gdf, :fpc2 = length(:county))
county_sizes = Dict(zip(cs[!, :county], cs[!, :fpc2]))

stage2_jl = @chain stage2 begin
    @groupby(:county)
    @combine(:subtotal = π_sum(:Burglary, county_sizes[first(:county)]))
    @combine(:total = π_sum(:subtotal, N_counties))
end

stage2_joined = innerjoin(stage2, cs, on=:county)
stage2_joined[:, :id] = 1:size(stage2_joined, 1)
@rput stage2_joined
stage2_r = R"svytotal(~Burglary, svydesign(id=~county+id, fpc=~fpc+fpc2, data=stage2_joined))"
r_comp(stage2_jl, stage2_r)
