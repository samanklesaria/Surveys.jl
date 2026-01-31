using Surveys, RCall, DataFrames, DataFramesMeta, Test, CSV, StatsBase

function run_tests()
    R"""
    library(survey)
    data(api)

    with_var <- function(result) {
        c(result, SE(result)^2)
    }
    """

    function r_comp(julia_val, rv)
        j = julia_val[1, :total]
        @test all(rv .≈ [j.sum, j.var])
    end

    # Simple Random Sampling

    @rget apisrs
    R"srs_design <- svydesign(id=~1, fpc=~fpc, data=apisrs)"
    r_comp(
        @combine(apisrs, :total = π_sum(:enroll, Int(:fpc[1]))),
        rcopy(rcall(:with_var, R"svytotal(~enroll, srs_design)")))

    # Stratified Sampling

    @rget apistrat
    strat_jl = @chain apistrat begin
        @groupby(:stype)
        @combine(:subtotal = π_sum(:enroll, Int(:fpc[1])))
        @combine(:total = sum(:subtotal))
    end
    R"strat_design <- svydesign(id=~1, fpc=~fpc, strata=~stype, data=apistrat)"
    r_comp(strat_jl, rcopy(rcall(:with_var, R"svytotal(~enroll, strat_design)")))

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
    r_comp(cluster1_jl, rcopy(rcall(:with_var, cluster1_r)))

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
    r_comp(stage2_jl, rcopy(rcall(:with_var, stage2_r)))

    # Ratio estimation
    ratio_r = last.(collect(rcopy(R"svyratio(~api.stu, ~enroll, srs_design)")))
    ratio_jl = @combine(apisrs, :total =
        π_sum((a -> a[1] / a[2]), [:api_stu :enroll], Int(:fpc[1])))
    r_comp(ratio_jl, ratio_r)

    # Stratified Taylor Approximations
    ratio_jl = @chain apistrat begin
        @groupby(:stype)
        @combine(:subtotal=π_sum([:api_stu :enroll],  Int(:fpc[1])))
        @combine(:total=sum(a->a[1] / a[2], :subtotal))
    end
    ratio_r = last.(collect(rcopy(R"svyratio(~api.stu, ~enroll, strat_design)")))
    r_comp(ratio_jl, ratio_r)

    # Clustered Taylor Approximations
    cluster2_jl = @chain stage2 begin
        @groupby(:county)
        @combine(:subtotal = π_sum([:Burglary :Murder], county_sizes[first(:county)]))
        @combine(:total = π_sum((a -> a[1] / a[2]), :subtotal, N_counties))
    end
    cluster2_r = last.(collect(rcopy(R"svyratio(~Burglary, ~Murder, svydesign(id=~county+id, fpc=~fpc+fpc2, data=stage2_joined))")))
    r_comp(cluster2_jl, cluster2_r)
end

run_tests()
