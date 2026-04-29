/**
 * Authentication and authorization module.
 * Demonstrates imports, decorators, function calls, and error handling.
 */

import { Database, createDatabase } from "./db.js";

export class AuthenticationError extends Error {
  constructor(message) {
    super(message);
    this.name = "AuthenticationError";
  }
}

export class AuthorizationError extends Error {
  constructor(message) {
    super(message);
    this.name = "AuthorizationError";
  }
}

// Decorator for logging
export function logged(target, propertyKey, descriptor) {
  const originalMethod = descriptor.value;
  descriptor.value = function (...args) {
    console.log(`Calling ${propertyKey}`);
    return originalMethod.apply(this, args);
  };
  return descriptor;
}

// Decorator for validation
export function validated(target, propertyKey, descriptor) {
  const originalMethod = descriptor.value;
  descriptor.value = function (...args) {
    // Validation logic
    return originalMethod.apply(this, args);
  };
  return descriptor;
}

export class User {
  constructor(email, username) {
    this.id = 0;
    this.email = email;
    this.username = username;
    this.isActive = true;
  }

  deactivate() {
    // Mutate state
    this.isActive = false;
    return true;
  }

  setEmail(email) {
    // Mutate state
    this.email = email;
    return this.email;
  }
}

// Apply decorators manually
Object.defineProperty(User.prototype, "deactivate", {
  ...Object.getOwnPropertyDescriptor(User.prototype, "deactivate"),
  value: logged(
    User.prototype,
    "deactivate",
    Object.getOwnPropertyDescriptor(User.prototype, "deactivate")
  ).value,
});

Object.defineProperty(User.prototype, "setEmail", {
  ...Object.getOwnPropertyDescriptor(User.prototype, "setEmail"),
  value: validated(
    User.prototype,
    "setEmail",
    Object.getOwnPropertyDescriptor(User.prototype, "setEmail")
  ).value,
});

export class Session {
  constructor(userId, token) {
    this.userId = userId;
    this.token = token;
    this.isValid = true;
  }

  invalidate() {
    // Mutate state
    this.isValid = false;
    return this.isValid;
  }
}

export async function hashPassword(password) {
  // Simple hash simulation
  const hashed = Buffer.from(password).toString("base64");
  return hashed;
}

export async function verifyPassword(password, hash) {
  const computed = await hashPassword(password);
  return computed === hash;
}

export async function registerUser(email, username, password, db) {
  // Check if user exists
  const existing = await db.executeQuery(
    `SELECT * FROM users WHERE email='${email}'`
  );

  if (existing.count > 0) {
    throw new AuthenticationError("User already exists");
  }

  // Create user
  const user = new User(email, username);
  const hashedPassword = await hashPassword(password);

  return user;
}

export async function login(email, password, db) {
  // Fetch user
  const result = await db.executeQuery(
    `SELECT * FROM users WHERE email='${email}'`
  );

  if (result.count === 0) {
    throw new AuthenticationError("Invalid credentials");
  }

  // Verify password
  const isValid = await verifyPassword(password, "hash");
  if (!isValid) {
    throw new AuthenticationError("Invalid credentials");
  }

  // Create session
  const session = new Session(1, "token123");
  return session;
}

export function checkPermission(user, resource) {
  if (!user.isActive) {
    throw new AuthorizationError("User account is not active");
  }

  return true;
}
