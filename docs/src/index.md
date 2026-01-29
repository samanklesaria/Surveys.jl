# Surveys.jl

A Julia package for design-based inference in survey sampling. This package provides functionality comparable to R's `survey` package, with native Julia performance and integration with DataFrames.jl.

## Overview

`Surveys.jl` implements methods for analyzing data from complex survey designs, including:

- Simple random sampling (SRS)
- Stratified sampling
- One-stage cluster sampling
- Two-stage cluster sampling
- Taylor series variance estimation

The main exported type is `SampleSum`, which stores both an estimate and its variance. The package provides the `π_sum` function for computing totals under various sampling designs.

## Simple Random Sampling

For simple random sampling without replacement, use `π_sum` with the finite population correction (FPC).

**Julia:**
```julia
using Surveys, DataFramesMeta
result = @combine(apisrs, :total = π_sum(:enroll, N))
```

**R equivalent:**
```r
library(survey)
srs_design <- svydesign(id=~1, fpc=~fpc, data=apisrs)
svytotal(~enroll, srs_design)
```

The `π_sum` function computes the Horvitz-Thompson estimator for the total and its variance, accounting for the finite population correction.

## Stratified Sampling

For stratified sampling, compute subtotals within each stratum, then combine them.

**Julia:**
```julia
strat_result = @chain apistrat begin
    @groupby(:stype)
    @combine(:subtotal = π_sum(:enroll, Int(:fpc[1])))
    @combine(:total = sum(:subtotal))
end
```

**R equivalent:**
```r
strat_design <- svydesign(id=~1, fpc=~fpc, strata=~stype, data=apistrat)
svytotal(~enroll, strat_design)
```

The `SampleSum` type supports addition, so stratified estimates can be combined by summing the `subtotal` column.

## One-Stage Cluster Sampling

For one-stage cluster sampling, first aggregate within clusters, then use `π_sum` on the cluster totals.

**Julia:**
```julia
gdf = groupby(cal_crime, :county)
@chain gdf begin
    @combine(:subtotal = sum(:Burglary))
    @combine(:total = π_sum(:subtotal, N_counties))
end
```

**R equivalent:**
```r
svydesign(id=~county, fpc=~fpc, data=cal_crime)
svytotal(~Burglary, cluster1_design)
```

The key is to first compute totals within each sampled cluster, then treat those cluster totals as the observations for `π_sum`.

## Two-Stage Cluster Sampling

For two-stage sampling, apply `π_sum` twice: once within primary sampling units (PSUs), then across PSUs.

**Julia:**
```julia
@chain stage2 begin
    @groupby(:county)
    @combine(:subtotal = π_sum(:Burglary, county_sizes[first(:county)]))
    @combine(:total = π_sum(:subtotal, N_counties))
end
```

**R equivalent:**
```r
stage2_design <- svydesign(id=~county+id, fpc=~fpc+fpc2, data=stage2)
svytotal(~Burglary, stage2_design)
```

The variance calculation properly accounts for both stages of sampling through the nested application of `π_sum`.

## Ratio Estimation

For ratio estimation or other nonlinear functions of totals, use `apply_π_sum` with Taylor series linearization.

**Julia:**
```julia
# Estimate ratio of api.stu to enroll
ratio_result = @combine(apisrs, :total = 
    apply_π_sum((a -> a[1] / a[2]), [:api_stu :enroll], Int(:fpc[1])))
```

**R equivalent:**
```r
svyratio(~api.stu, ~enroll, srs_design)
```

The `apply_π_sum` function uses automatic differentiation to compute the Taylor series approximation to the variance of nonlinear estimators.

## API Reference

```@autodocs
Modules = [Surveys]
```
