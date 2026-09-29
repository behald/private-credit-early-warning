select
    lender_cik,
    bdc_name,
    quarter,
    count(distinct canonical_borrower_id) as portfolio_size,
    avg(mark) as avg_mark,
    avg(mark_drift) as avg_mark_drift,
    stddev(mark) as mark_stddev,
    sum(case when non_accrual_flag = 1 then 1 else 0 end) as non_accrual_count,
    sum(case when pik_flag = 1 then 1 else 0 end) as pik_count,
    sum(case when mark < 0.95 then 1 else 0 end) as below_par_count,
    sum(case when mark < 0.80 then 1 else 0 end) as deeply_distressed_count,
    sum(fair_value) as total_fair_value,
    sum(cost) as total_cost,
    sum(fair_value) / nullif(sum(cost), 0) as portfolio_weighted_mark
from {{ ref('mart_signals') }}
group by lender_cik, bdc_name, quarter
