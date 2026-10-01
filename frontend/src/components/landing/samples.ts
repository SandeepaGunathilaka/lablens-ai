import type { ReportRow } from './ReportSheet'

// Illustrative values only, not real patient data.
export const CBC_SAMPLE: ReportRow[] = [
  { test: 'Hemoglobin', result: '10.2', unit: 'g/dL', reference: '12.0–15.5', flag: 'low' },
  { test: 'WBC', result: '7.0', unit: '×10⁹/L', reference: '4.0–11.0' },
  { test: 'Platelets', result: '250', unit: '×10⁹/L', reference: '150–450' },
]

export const LIPID_SAMPLE: ReportRow[] = [
  { test: 'Total Cholesterol', result: '180', unit: 'mg/dL', reference: '< 200' },
  { test: 'LDL', result: '160', unit: 'mg/dL', reference: '< 100', flag: 'high' },
  { test: 'HDL', result: '35', unit: 'mg/dL', reference: '≥ 40', flag: 'low' },
]
