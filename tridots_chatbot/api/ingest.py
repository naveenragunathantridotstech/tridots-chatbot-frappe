import os
import sys
import frappe
from tridots_chatbot.ingest.scrape import run_scrape_pipeline
from tridots_chatbot.ingest.run import main as run_embedding_pipeline

def get_site_paths():
    """Resolves site-specific folders dynamically using Frappe standard API"""
    private_files_dir = frappe.get_site_path("private", "files")
    
    # Ensure folders exist
    scraped_content_dir = os.path.join(private_files_dir, "scraped_content")
    os.makedirs(scraped_content_dir, exist_ok=True)
    
    vectors_path = os.path.join(private_files_dir, "vectors.json")
    manifest_path = os.path.join(private_files_dir, "scrape_manifest.json")
    sitemap_path = "https://www.tridotstech.com/sitemap.xml"
    
    return sitemap_path, scraped_content_dir, manifest_path, vectors_path

@frappe.whitelist()
def run_initial_ingestion():
    """Runs automatically after the app is installed for the first time"""
    frappe.enqueue(
        "tridots_chatbot.api.ingest.execute_ingestion",
        queue="long",
        timeout=1200,
        now=frappe.flags.in_test
    )

def run_weekly_ingestion():
    """Runs automatically on a weekly basis via the Frappe scheduler"""
    frappe.enqueue(
        "tridots_chatbot.api.ingest.execute_ingestion",
        queue="long",
        timeout=1200,
        kwargs={"incremental": True}
    )

def execute_ingestion(incremental=False):
    """Downloads sitemap, scrapes modified/new pages, and updates embeddings incrementally"""
    sitemap_path, scraped_content_dir, manifest_path, vectors_path = get_site_paths()
    
    frappe.logger().info("Starting Chatbot scraping pipeline...")
    
    # 1. Run Scraper
    run_scrape_pipeline(
        sitemap_path=sitemap_path,
        output_dir=scraped_content_dir,
        manifest_path=manifest_path,
        incremental=incremental
    )
    
    frappe.logger().info("Scraping completed. Running embeddings generation...")
    
    # 2. Run Embedding Generator
    args = [
        "--input-dir", scraped_content_dir,
        "--output", vectors_path
    ]
    if incremental:
        args.append("--incremental")
        
    run_embedding_pipeline(args)
    
    frappe.logger().info(f"Chatbot embeddings successfully updated at {vectors_path}.")
