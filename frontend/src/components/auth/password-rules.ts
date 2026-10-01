// Mirrors RegisterRequest in backend/security/auth.py so users see problems
// before submitting. The backend still validates; these are only hints.
export const MIN_PASSWORD_LENGTH = 8
export const BCRYPT_MAX_BYTES = 72

export function passwordChecks(password: string) {
  return [
    { label: `At least ${MIN_PASSWORD_LENGTH} characters`, met: password.length >= MIN_PASSWORD_LENGTH },
    { label: 'Not only spaces', met: password.trim().length > 0 },
    {
      label: `No longer than ${BCRYPT_MAX_BYTES} bytes`,
      met: password.length > 0 && new TextEncoder().encode(password).length <= BCRYPT_MAX_BYTES,
    },
  ]
}
