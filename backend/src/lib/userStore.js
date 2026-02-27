/**
 * Persistent user storage.
 */
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, '../../data');
const USERS_FILE = path.join(DATA_DIR, 'users.json');

let users = [];

async function ensureDataDir() {
  await fs.mkdir(DATA_DIR, { recursive: true });
}

async function load() {
  try {
    const data = await fs.readFile(USERS_FILE, 'utf-8');
    users = JSON.parse(data);
  } catch {
    users = [];
  }
}

async function save() {
  await ensureDataDir();
  await fs.writeFile(USERS_FILE, JSON.stringify(users, null, 2));
}

export async function findUserByEmail(email) {
  await load();
  return users.find((u) => u.email.toLowerCase() === email.toLowerCase());
}

export async function findUserById(id) {
  await load();
  return users.find((u) => u.id === id);
}

export async function createUser(user) {
  await load();
  users.push(user);
  await save();
  return user;
}

export async function updateUser(id, updates) {
  await load();
  const idx = users.findIndex((u) => u.id === id);
  if (idx === -1) return null;
  users[idx] = { ...users[idx], ...updates };
  await save();
  return users[idx];
}
