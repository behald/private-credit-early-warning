with panel as (
    select *,
        fair_value / nullif(cost, 0) as mark,
        row_number() over (
            partition by canonical_borrower_id, lender_cik
            order by quarter
        ) as quarter_seq
    from {{ ref('int_quarterly_panel') }}
),

with_lag as (
    select p.*,
        lag(p.mark) over (
            partition by p.canonical_borrower_id, p.lender_cik
            order by p.quarter
        ) as prior_mark,
        lag(p.pik_flag, 1, 0) over (
            partition by p.canonical_borrower_id, p.lender_cik
            order by p.quarter
        ) as prior_pik_flag,
        lag(p.non_accrual_flag, 1, 0) over (
            partition by p.canonical_borrower_id, p.lender_cik
            order by p.quarter
        ) as prior_non_accrual_flag
    from panel p
),

cross_lender as (
    select
        canonical_borrower_id,
        quarter,
        count(distinct lender_cik) as num_lenders,
        avg(mark) as avg_mark_across_lenders,
        stddev(mark) as mark_dispersion,
        min(mark) as min_mark,
        max(mark) as max_mark
    from panel
    where mark is not null
    group by canonical_borrower_id, quarter
)

select
    wl.canonical_borrower_id,
    wl.lender_cik,
    wl.bdc_name,
    wl.quarter,
    wl.fair_value,
    wl.cost,
    wl.mark,
    wl.prior_mark,
    wl.mark - wl.prior_mark as mark_drift,
    wl.pik_flag,
    wl.non_accrual_flag,
    case when wl.prior_pik_flag = 0 and wl.pik_flag = 1 then 1 else 0 end as pik_migration,
    case when wl.prior_mark >= 0.95 and wl.mark < 0.95 then 1 else 0 end as par_migration,
    case when wl.prior_non_accrual_flag = 0 and wl.non_accrual_flag = 1 then 1 else 0 end as non_accrual_migration,
    cl.num_lenders,
    cl.avg_mark_across_lenders,
    cl.mark_dispersion,
    cl.min_mark,
    cl.max_mark,
    wl.mark - cl.avg_mark_across_lenders as lender_mark_vs_avg,
    wl.industry,
    wl.position_count,
    wl.quarter_seq
from with_lag wl
left join cross_lender cl
    on wl.canonical_borrower_id = cl.canonical_borrower_id
    and wl.quarter = cl.quarter
