SELECT 
    -- Features (X)
    loan_amnt,
    term,
    int_rate,
    installment,
    sub_grade,
    purpose,
    addr_state,
    annual_inc,
    home_ownership,
    emp_length,
    verification_status,
    COALESCE(dti_joint, dti) AS effective_dti,
    CASE WHEN application_type = 'Joint App' THEN 1 ELSE 0 END AS is_joint,
    fico_range_low AS fico_score,
    revol_util,
    revol_bal,
    open_acc,
    total_acc,
    inq_last_6mths,
    delinq_2yrs,
    pub_rec,
    pub_rec_bankruptcies,
    mort_acc,
    earliest_cr_line,
    issue_d,
    
    -- Target (y)
    loan_status
FROM loans
TABLESAMPLE BERNOULLI (1)
LIMIT 10000;
