--line count
SELECT COUNT(*) FROM loans;

--quick check
SELECT 
    id,
    loan_amt,
    term,
    int_rate, 
    grade, 
    loan_status, 
    annual_inc, 
    dti
FROM loans
LIMIT 10;

--null percentages

--outcome percentages


