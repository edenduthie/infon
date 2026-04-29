/**
 * Main application entry point.
 * Demonstrates all eight relation types in JavaScript.
 */

import {
  Database,
  createDatabase,
  DatabaseError,
  ConnectionError,
  QueryError,
} from "./db.js";
import {
  User,
  Session,
  registerUser,
  login,
  checkPermission,
  AuthenticationError,
  AuthorizationError,
} from "./auth.js";

async function initializeDatabase() {
  const config = {
    host: "localhost",
    port: 5432,
    database: "app_db",
  };

  const db = createDatabase(config);
  await db.connect();
  return db;
}

async function runQuery(db, query) {
  try {
    const result = await db.executeQuery(query);
    return result.rows;
  } catch (error) {
    if (error instanceof QueryError) {
      console.error(`Query error: ${error.message}`);
      throw error;
    }
    if (error instanceof ConnectionError) {
      console.error(`Connection error: ${error.message}`);
      throw error;
    }
    throw error;
  }
}

async function handleUserRegistration(email, username, password) {
  const db = await initializeDatabase();

  try {
    const user = await registerUser(email, username, password, db);
    return user;
  } catch (error) {
    if (error instanceof AuthenticationError) {
      console.error(`Registration failed: ${error.message}`);
    }
    return null;
  } finally {
    await db.disconnect();
  }
}

async function handleUserLogin(email, password) {
  const db = await initializeDatabase();

  try {
    const session = await login(email, password, db);
    return session;
  } catch (error) {
    if (error instanceof AuthenticationError) {
      console.error(`Login failed: ${error.message}`);
    }
    return null;
  } finally {
    await db.disconnect();
  }
}

async function main() {
  // Register user
  const user = await handleUserRegistration(
    "user@example.com",
    "testuser",
    "password123"
  );

  if (user) {
    // Login
    const session = await handleUserLogin("user@example.com", "password123");

    if (session) {
      try {
        // Check permissions
        const hasAccess = checkPermission(user, "admin_panel");
        console.log(`Access granted: ${hasAccess}`);
      } catch (error) {
        if (error instanceof AuthorizationError) {
          console.error(`Authorization error: ${error.message}`);
        }
      }

      // Logout
      session.invalidate();
    }
  }

  return 0;
}

// Run main
main()
  .then((code) => {
    process.exit(code);
  })
  .catch((error) => {
    console.error(`Fatal error: ${error}`);
    process.exit(1);
  });
