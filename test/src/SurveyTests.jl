module SurveyTests

using Surveys, RCall, DataFrames, DataFramesMeta, CSV, StatsBase, StatsModels, GLM
using ReTest

r_comp(julia_val::DataFrame, rv) = r_comp(julia_val[1, :total], rv)
r_comp(j::SampleSum, rv) = @test all(rv .≈ [j.sum, j.var])

@testset "R comparison" begin

    # Utility Functions
    R"""
    library(sampling)
    library(survey)
    data(api)

    with_var <- function(result) {
        c(result, SE(result)^2)
    }
    """

    # Getting the data
    cal_crime = DataFrame(CSV.File("test/crime_data.csv"))
    @rget apisrs
    @rget apistrat

    @testset "Simple Random Sampling" begin
        R"srs_design <- svydesign(id=~1, fpc=~fpc, data=apisrs)"
        r_comp(
            @combine(apisrs, :total = sum(:enroll, SI(Int(:fpc[1])))),
            rcopy(rcall(:with_var, R"svytotal(~enroll, srs_design)")))
    end

    @testset "Stratified Sampling" begin
        strat_jl = @chain apistrat begin
            @groupby(:stype)
            @combine(:subtotal = sum(:enroll, SI(Int(:fpc[1]))))
            @combine(:total = sum(:subtotal))
        end
        R"strat_design <- svydesign(id=~1, fpc=~fpc, strata=~stype, data=apistrat)"
        r_comp(strat_jl, rcopy(rcall(:with_var, R"svytotal(~enroll, strat_design)")))
    end

    @testset "Cluster Sampling" begin
        counties = unique(cal_crime[!, :county])
        chosen_counties = Set(sample(counties, 10, replace=false))

        crime_1stage = @chain cal_crime begin
            @subset(:county .∈ Ref(chosen_counties))
            @groupby(:county)
            @transform(:fpc = length(counties))
        end
        grouped_crime = groupby(crime_1stage, :county)
        @rput crime_1stage

        @testset "One Stage Sampling" begin
            cluster1_jl = @chain grouped_crime begin
                @combine(:subtotal = sum(:Burglary))
                @combine(:total = sum(:subtotal, SI(length(counties))))
            end
            cluster1_r = R"svytotal(~Burglary, svydesign(id=~county, fpc=~fpc, data=crime_1stage))"
            r_comp(cluster1_jl, rcopy(rcall(:with_var, cluster1_r)))
        end

        @testset "Two Stage Sampling" begin
            crime_2stage = combine(grouped_crime, g -> g[sample(1:size(g, 1),
                min(size(g, 1), 5); replace=false), :])
            cs = @combine(grouped_crime, :fpc2 = length(:county))
            county_sizes = Dict(zip(cs[!, :county], cs[!, :fpc2]))

            @testset "Cluster Sums" begin
                stage2_jl = @chain crime_2stage begin
                    @groupby(:county)
                    @combine(:subtotal = sum(:Burglary, SI(county_sizes[first(:county)])))
                    @combine(:total = sum(:subtotal, SI(length(counties))))
                end

                stage2_joined = innerjoin(crime_2stage, cs, on=:county)
                stage2_joined[:, :id] = 1:size(stage2_joined, 1)
                @rput stage2_joined
                stage2_r = R"svytotal(~Burglary, svydesign(id=~county+id, fpc=~fpc+fpc2, data=stage2_joined))"
                r_comp(stage2_jl, rcopy(rcall(:with_var, stage2_r)))
            end

            @testset "Clustered Taylor Approximations" begin
                cluster2_jl = @chain crime_2stage begin
                    @groupby(:county)
                    @combine(:subtotal = sum([:Burglary :Theft], SI(county_sizes[first(:county)])))
                    @combine(:total = sum((a -> a[1] / a[2]), :subtotal, SI(length(counties))))
                end
                cluster2_r = last.(collect(rcopy(R"svyratio(~Burglary, ~Theft, svydesign(id=~county+id, fpc=~fpc+fpc2, data=stage2_joined))")))
                r_comp(cluster2_jl, cluster2_r)
            end
        end
    end

    @testset "Ratio Estimation" begin
        ratio_r = last.(collect(rcopy(R"svyratio(~api.stu, ~enroll, srs_design)")))
        ratio_jl = @combine(apisrs, :total =
            sum((a -> a[1] / a[2]), [:api_stu :enroll], SI(Int(:fpc[1]))))
        r_comp(ratio_jl, ratio_r)
    end

    @testset "Stratified Taylor Approximations" begin
        ratio_jl = @chain apistrat begin
            @groupby(:stype)
            @combine(:subtotal = sum([:api_stu :enroll], SI(Int(:fpc[1]))))
            @combine(:total = sum(a -> a[1] / a[2], :subtotal))
        end
        ratio_r = last.(collect(rcopy(R"svyratio(~api.stu, ~enroll, strat_design)")))
        r_comp(ratio_jl, ratio_r)
    end

    @testset "Coefficient Estimation" begin
        R"model <- svyglm(api.stu ~ enroll, srs_design)"
        j = lm(@formula(api_stu ~ 1 + enroll), apisrs, SI(Int(apisrs[1, :fpc])))
        @test all(rcopy(R"coef(model)") .≈ [a.sum for a in j])
        @test all(rcopy(R"SE(model)^2") .≈ [a.var for a in j])
    end

    @testset "Model Assisted Estimation" begin
        assisted_jl = sum(@formula(api_stu ~ 1 + enroll), apisrs, (; enroll=[4e6]), SI(Int(apisrs[1, :fpc])))
        assisted_r = R"svytotal(~api.stu, calibrate(srs_design, ~enroll, c('(Intercept)'=6194, enroll=4e6)))"
        r_comp(assisted_jl, rcopy(rcall(:with_var, assisted_r)))
    end

    @testset "PPS" begin
        @testset "Without Replacement" begin
            R"pik <- inclusionprobabilities($(cal_crime[!, :Theft]), 50)"
            R"pi2 <- UPmaxentropypi2(pik)"
            @rget pik pi2
            s = sample(1:size(cal_crime, 1), Weights(pik), 50; replace=false)
            crime_pps = cal_crime[s, :]
            pik = pik[s]
            pi2 = pi2[s, s]
            @transform!(crime_pps, :probs = pik)
            @rput crime_pps
            pps1_r = rcopy(rcall(:with_var,
                R"""svytotal(~Burglary,svydesign(id=~1,
                probs=~probs, pps=ppsmat($pi2), data=crime_pps))"""))
            pps1_jl = sum(cal_crime[s, :Burglary], WithoutReplacement(pik, pi2))
            r_comp(pps1_jl, pps1_r)
        end
        @testset "With Replacement" begin
            weights = Weights(cal_crime[!, :Theft])
            samples = sample(1:size(cal_crime, 1), weights, 92; replace=true)
            crime_pps = cal_crime[samples, :]
            probs = (weights./weights.sum)[samples]
            @transform!(crime_pps, :probs = probs .* length(samples))
            @rput crime_pps
            pps1_r = rcopy(rcall(:with_var,
                R"svytotal(~Burglary, svydesign(id=~1, probs=~probs, data=crime_pps))"))
            pps1_jl = sum(crime_pps[!, :Burglary], WithReplacement(probs))
            r_comp(pps1_jl, pps1_r)
        end
    end
end

end
