/**
 * Database connection and query management.
 * Demonstrates classes, interfaces, errors, and mutations.
 */

export class DatabaseError extends Error {
  constructor(message) {
    super(message);
    this.name = "DatabaseError";
  }
}

export class ConnectionError extends DatabaseError {
  constructor(message) {
    super(message);
    this.name = "ConnectionError";
  }
}

export class QueryError extends DatabaseError {
  constructor(message) {
    super(message);
    this.name = "QueryError";
  }
}

export class Database {
  constructor(config) {
    this.host = config.host;
    this.port = config.port;
    this.database = config.database;
    this.connected = false;
    this.connection = null;
  }

  async connect() {
    if (this.connected) {
      return true;
    }

    try {
      // Mutate state
      this.connected = true;
      this.connection = { host: this.host, port: this.port };
      return true;
    } catch (error) {
      throw new ConnectionError(`Failed to connect: ${error}`);
    }
  }

  async disconnect() {
    // Mutate state
    this.connected = false;
    this.connection = null;
  }

  async executeQuery(query) {
    if (!this.connected) {
      throw new QueryError("Not connected to database");
    }

    // Simulate query execution
    const result = {
      rows: [],
      count: 0,
    };
    return result;
  }

  getConnectionInfo() {
    return {
      host: this.host,
      port: this.port,
      database: this.database,
      connected: this.connected,
    };
  }
}

export function createDatabase(config) {
  const db = new Database(config);
  return db;
}
