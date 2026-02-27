/**
 * Persistent storage for feedback reports.
 * Transcripts are never stored - only the AI-generated feedback report.
 */
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, '../../data');
const REPORTS_FILE = path.join(DATA_DIR, 'feedback-reports.json');

let reports = [];

async function ensureDataDir() {
  await fs.mkdir(DATA_DIR, { recursive: true });
}

async function load() {
  try {
    const data = await fs.readFile(REPORTS_FILE, 'utf-8');
    reports = JSON.parse(data);
  } catch {
    reports = [];
  }
}

async function save() {
  await ensureDataDir();
  await fs.writeFile(REPORTS_FILE, JSON.stringify(reports, null, 2));
}

export async function addReport(report) {
  await load();
  reports.push(report);
  await save();
  return report;
}

export async function getReportsByUserId(userId) {
  await load();
  return reports
    .filter((r) => r.userId === userId)
    .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
}

export async function getReportById(id) {
  await load();
  return reports.find((r) => r.id === id);
}
