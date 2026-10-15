# Review of 12 actual classifier mistakes

The final head makes 32 raw-label errors. The following 12 were reviewed from saved predictions, including errors rejected by the confidence gate. Causes are hypotheses based on wording.

| ticket_id | text | y_true | y_pred | accepted | cause | next_step |
|---|---|---|---|---|---|---|
| EN-023381 | help me to check the sxtatus of my reimbursement | track_refund | check_refund_policy | False | typos | Add reviewed typo examples and spelling variants. |
| EN-013284 | i have to get the bill #85632 will huelp me | get_invoice | check_invoice | True | intent_boundary | Add contrastive get_invoice/check_invoice examples. |
| AR-TEST-0010 | انتقلت إلى شقة جديدة، كيف أحدث عنوان الشحن للطلب الحالي؟ | change_shipping_address | set_up_shipping_address | False | intent_boundary | Add examples emphasizing current order versus saved address. |
| AR-TEST-0013 | فيه غرامة لو كنسلت الطلب؟ | check_cancellation_fee | complaint | False | dialect_coverage | Collect varied Gulf cancellation-fee messages. |
| AR-TEST-0025 | كم يوم عندي عشان ارجع المنتج؟ | check_refund_policy | delivery_period | False | intent_boundary | Add contrasts between return window and delivery estimate. |
| AR-TEST-0026 | هل يمكن استرداد المبلغ للمنتجات المخفضة؟ | check_refund_policy | get_refund | False | intent_boundary | Define the policy-versus-action distinction with paired examples. |
| AR-TEST-0035 | وين القى طريقة اتواصل معكم؟ | contact_customer_service | contact_human_agent | False | ambiguous_wording | Add contact-channel examples and ask a clarification when needed. |
| AR-TEST-0042 | ما هي خطوات التسجيل في الموقع؟ | create_account | registration_problems | False | intent_boundary | Add successful signup questions contrasted with signup failures. |
| AR-TEST-0061 | ابي الفاتورة الضريبية للطلب A-77120 | get_invoice | track_order | False | dialect_coverage | Add short Gulf get_invoice examples with identifiers. |
| AR-TEST-0073 | الدفع ما ضبط بس انخصم المبلغ من حسابي | payment_issue | get_refund | False | dialect_coverage | Collect payment-failure phrases in Gulf dialect. |
| AR-TEST-0081 | نسيت الباسوورد ابي استرجعه | recover_password | track_refund | False | transliteration | Add Arabic spellings of English password terminology. |
| AR-TEST-0105 | وين وصل استرجاع فلوسي؟ | track_refund | recover_password | False | intent_boundary | Add refund-tracking utterances with money and timing context. |

Counts: {'intent_boundary': 6, 'dialect_coverage': 3, 'typos': 1, 'ambiguous_wording': 1, 'transliteration': 1}

The only accepted error in these 12 reviewed cases is EN-013284, an invoice request routed to check_invoice. Abstention protects many examples but does not catch every mistake.
