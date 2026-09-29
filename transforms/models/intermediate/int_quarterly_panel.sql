with positions as (
    select * from {{ ref('stg_soi_positions') }}
),

panel as (
    select
        canonical_borrower_id,
        cik as lender_cik,
        bdc_name,
        quarter,
        sum(fair_value) as fair_value,
        sum(cost) as cost,
        max(case when investment_type like '%PIK%' then 1 else 0 end) as pik_flag,
        max(case when investment_type like '%NonAccrual%'
                  or investment_type like '%Non-Accrual%'
                  or investment_type like '%non_accrual%' then 1 else 0 end) as non_accrual_flag,
        max(industry) as industry,
        count(*) as position_count
    from positions
    group by canonical_borrower_id, cik, bdc_name, quarter
)

select * from panel
