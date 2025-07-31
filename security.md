

## Secure Development Knowledge

At least one core developer is familiar with common Python security vulnerabilities, including:

- Code injection and deserialization risks
- Command and path injection
- Dependency and secret management
- Secure authentication and cryptographic practices

We actively apply mitigation strategies such as input validation, secure coding patterns, and dependency auditing to reduce risk.

## Use of Cryptography


This project is not a cryptographic library. When cryptographic functionality is required (e.g., hashing, encryption), we rely exclusively on well-established, FLOSS cryptographic libraries such as:

- `cryptography`
- `hashlib`
- `secrets`

We do not implement custom cryptographic algorithms or primitives. This ensures our use of cryptography is secure, reliable, and aligned with best practices.


This project uses cryptographic functions (e.g., SHA-256) for file integrity checking. These functions are implemented using Python’s standard `hashlib` library, which is a well-established and FLOSS cryptographic tool. We do not implement custom cryptographic algorithms.
