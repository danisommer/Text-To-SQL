#!/usr/bin/env python3
"""
Text-to-SQL Database Query System
A comprehensive tool for converting natural language queries to SQL
and executing them on MySQL and PostgreSQL databases.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import re

# Database connectors
try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import psycopg2
    from psycopg2 import sql, Error as PostgreSQLError
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

# Text-to-SQL libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Other dependencies
import sqlite3
from tabulate import tabulate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str
    db_type: str  # 'mysql' or 'postgresql'

@dataclass
class TableInfo:
    """Table information structure"""
    name: str
    columns: List[Dict[str, str]]
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]

class DatabaseManager:
    """Handles database connections and operations"""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.db_type = None
        self.config = None
    
    def connect(self, config: DatabaseConfig) -> bool:
        """Connect to database"""
        try:
            self.config = config
            self.db_type = config.db_type.lower()
            
            if self.db_type == 'mysql':
                if not MYSQL_AVAILABLE:
                    logger.error("MySQL connector not available. Install with: pip install mysql-connector-python")
                    return False
                
                self.connection = mysql.connector.connect(
                    host=config.host,
                    port=config.port,
                    database=config.database,
                    user=config.username,
                    password=config.password
                )
                self.cursor = self.connection.cursor(dictionary=True)
                
            elif self.db_type == 'postgresql':
                if not POSTGRESQL_AVAILABLE:
                    logger.error("PostgreSQL connector not available. Install with: pip install psycopg2-binary")
                    return False
                
                self.connection = psycopg2.connect(
                    host=config.host,
                    port=config.port,
                    database=config.database,
                    user=config.username,
                    password=config.password
                )
                self.cursor = self.connection.cursor()
                
            else:
                logger.error(f"Unsupported database type: {config.db_type}")
                return False
            
            logger.info(f"Connected to {config.db_type} database: {config.database}")
            return True
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from database"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
    
    def get_tables(self) -> List[str]:
        """Get list of tables in the database"""
        try:
            if self.db_type == 'mysql':
                self.cursor.execute("SHOW TABLES")
                tables = [row[f'Tables_in_{self.config.database}'] for row in self.cursor.fetchall()]
            elif self.db_type == 'postgresql':
                self.cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                tables = [row[0] for row in self.cursor.fetchall()]
            
            return tables
        except Exception as e:
            logger.error(f"Error getting tables: {e}")
            return []
    
    def get_table_info(self, table_name: str) -> Optional[TableInfo]:
        """Get detailed information about a table"""
        try:
            columns = []
            primary_keys = []
            foreign_keys = []
            
            if self.db_type == 'mysql':
                # Get column information
                self.cursor.execute(f"DESCRIBE {table_name}")
                for row in self.cursor.fetchall():
                    columns.append({
                        'name': row['Field'],
                        'type': row['Type'],
                        'nullable': row['Null'] == 'YES',
                        'default': row['Default'],
                        'extra': row['Extra']
                    })
                    if row['Key'] == 'PRI':
                        primary_keys.append(row['Field'])
                
                # Get foreign key information
                self.cursor.execute(f"""
                    SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = '{self.config.database}'
                    AND TABLE_NAME = '{table_name}'
                    AND REFERENCED_TABLE_NAME IS NOT NULL
                """)
                for row in self.cursor.fetchall():
                    foreign_keys.append({
                        'column': row['COLUMN_NAME'],
                        'referenced_table': row['REFERENCED_TABLE_NAME'],
                        'referenced_column': row['REFERENCED_COLUMN_NAME']
                    })
            
            elif self.db_type == 'postgresql':
                # Get column information
                self.cursor.execute(f"""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position
                """)
                for row in self.cursor.fetchall():
                    columns.append({
                        'name': row[0],
                        'type': row[1],
                        'nullable': row[2] == 'YES',
                        'default': row[3],
                        'extra': ''
                    })
                
                # Get primary keys
                self.cursor.execute(f"""
                    SELECT column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.table_name = '{table_name}'
                    AND tc.constraint_type = 'PRIMARY KEY'
                """)
                primary_keys = [row[0] for row in self.cursor.fetchall()]
                
                # Get foreign keys
                self.cursor.execute(f"""
                    SELECT kcu.column_name, ccu.table_name, ccu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.table_name = '{table_name}'
                    AND tc.constraint_type = 'FOREIGN KEY'
                """)
                for row in self.cursor.fetchall():
                    foreign_keys.append({
                        'column': row[0],
                        'referenced_table': row[1],
                        'referenced_column': row[2]
                    })
            
            return TableInfo(
                name=table_name,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys
            )
            
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            return None
    
    def execute_query(self, query: str) -> Tuple[bool, List[Dict], str]:
        """Execute SQL query and return results"""
        try:
            self.cursor.execute(query)
            
            # Handle different query types
            if query.strip().lower().startswith(('select', 'show', 'describe', 'explain')):
                if self.db_type == 'mysql':
                    results = self.cursor.fetchall()
                elif self.db_type == 'postgresql':
                    columns = [desc[0] for desc in self.cursor.description]
                    rows = self.cursor.fetchall()
                    results = [dict(zip(columns, row)) for row in rows]
                
                return True, results, "Query executed successfully"
            else:
                self.connection.commit()
                affected_rows = self.cursor.rowcount
                return True, [], f"Query executed successfully. Affected rows: {affected_rows}"
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return False, [], str(e)
    
    def create_database(self, db_name: str) -> bool:
        """Create a new database"""
        try:
            if self.db_type == 'mysql':
                self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
            elif self.db_type == 'postgresql':
                # PostgreSQL requires autocommit for CREATE DATABASE
                self.connection.autocommit = True
                self.cursor.execute(f"CREATE DATABASE {db_name}")
                self.connection.autocommit = False
            
            self.connection.commit()
            logger.info(f"Database '{db_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database '{db_name}': {e}")
            return False

class TextToSQLConverter:
    """Handles conversion of natural language to SQL"""
    
    def __init__(self):
        self.method = None
        self.model = None
        self.tokenizer = None
        self.client = None
    
    def initialize_openai(self, api_key: str) -> bool:
        """Initialize OpenAI GPT for text-to-SQL conversion"""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI library not available. Install with: pip install openai")
            return False
        
        try:
            openai.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key)
            self.method = 'openai'
            logger.info("OpenAI GPT initialized for text-to-SQL conversion")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            return False
    
    def initialize_local_model(self, model_name: str = "microsoft/DialoGPT-medium") -> bool:
        """Initialize local transformer model for text-to-SQL"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available. Install with: pip install transformers torch")
            return False
        
        try:
            # For demo purposes, we'll use a simple approach
            # In production, you'd want to use specialized text-to-SQL models
            self.method = 'local'
            logger.info("Local model initialized for text-to-SQL conversion")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize local model: {e}")
            return False
    
    def convert_to_sql(self, natural_query: str, schema_info: Dict[str, TableInfo]) -> str:
        """Convert natural language query to SQL"""
        if self.method == 'openai':
            return self._convert_with_openai(natural_query, schema_info)
        elif self.method == 'local':
            return self._convert_with_local_model(natural_query, schema_info)
        else:
            return self._convert_with_rules(natural_query, schema_info)
    
    def _convert_with_openai(self, natural_query: str, schema_info: Dict[str, TableInfo]) -> str:
        """Convert using OpenAI GPT"""
        try:
            # Prepare schema context
            schema_context = self._prepare_schema_context(schema_info)
            
            prompt = f"""
            Given the following database schema:
            {schema_context}
            
            Convert this natural language query to SQL:
            "{natural_query}"
            
            Return only the SQL query, no explanations.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            # Clean up the response
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            return sql_query
            
        except Exception as e:
            logger.error(f"OpenAI conversion failed: {e}")
            return self._convert_with_rules(natural_query, schema_info)
    
    def _convert_with_local_model(self, natural_query: str, schema_info: Dict[str, TableInfo]) -> str:
        """Convert using local transformer model"""
        # For demonstration, we'll use rule-based conversion
        # In practice, you'd use a specialized text-to-SQL model
        return self._convert_with_rules(natural_query, schema_info)
    
    def _convert_with_rules(self, natural_query: str, schema_info: Dict[str, TableInfo]) -> str:
        """Simple rule-based conversion for demonstration"""
        query = natural_query.lower()
        
        # Basic patterns
        select_patterns = ['average', 'avg', 'mean', 'count', 'sum', 'max', 'min']
        table_names = list(schema_info.keys())
        
        # Try to identify the main operation
        operation = 'SELECT'
        if any(word in query for word in ['insert', 'add', 'create']):
            operation = 'INSERT'
        elif any(word in query for word in ['update', 'modify', 'change']):
            operation = 'UPDATE'
        elif any(word in query for word in ['delete', 'remove']):
            operation = 'DELETE'
        
        # Try to identify table
        table = None
        for table_name in table_names:
            if table_name.lower() in query:
                table = table_name
                break
        
        if not table and table_names:
            table = table_names[0]  # Default to first table
        
        # Build basic SELECT query
        if operation == 'SELECT' and table:
            if 'average' in query or 'avg' in query:
                # Look for numeric columns
                numeric_cols = []
                for col in schema_info[table].columns:
                    if any(t in col['type'].lower() for t in ['int', 'float', 'decimal', 'numeric']):
                        numeric_cols.append(col['name'])
                
                if numeric_cols:
                    col = numeric_cols[0]
                    sql = f"SELECT AVG({col}) FROM {table}"
                    
                    # Add WHERE clause if year is mentioned
                    year_match = re.search(r'\b(19|20)\d{2}\b', query)
                    if year_match:
                        year = year_match.group()
                        # Look for date columns
                        date_cols = [col['name'] for col in schema_info[table].columns 
                                   if any(t in col['type'].lower() for t in ['date', 'time', 'year'])]
                        if date_cols:
                            sql += f" WHERE YEAR({date_cols[0]}) = {year}"
                    
                    return sql
            
            elif 'count' in query:
                return f"SELECT COUNT(*) FROM {table}"
            
            else:
                # Default to SELECT all
                return f"SELECT * FROM {table} LIMIT 10"
        
        return f"SELECT * FROM {table if table else 'table_name'} LIMIT 10"
    
    def _prepare_schema_context(self, schema_info: Dict[str, TableInfo]) -> str:
        """Prepare schema information for AI context"""
        context = []
        for table_name, table_info in schema_info.items():
            cols = [f"{col['name']} ({col['type']})" for col in table_info.columns]
            context.append(f"Table: {table_name}\nColumns: {', '.join(cols)}")
        return "\n\n".join(context)

class TextToSQLApp:
    """Main application class"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.converter = TextToSQLConverter()
        self.schema_cache = {}
    
    def run(self):
        """Main application loop"""
        print("=" * 60)
        print("        TEXT-TO-SQL DATABASE QUERY SYSTEM")
        print("=" * 60)
        print()
        
        # Initialize text-to-SQL converter
        self._initialize_converter()
        
        # Main loop
        while True:
            try:
                self._show_menu()
                choice = input("\nEnter your choice: ").strip()
                
                if choice == '1':
                    self._connect_to_database()
                elif choice == '2':
                    self._show_tables()
                elif choice == '3':
                    self._natural_language_query()
                elif choice == '4':
                    self._execute_sql_query()
                elif choice == '5':
                    self._create_database()
                elif choice == '6':
                    self._show_table_details()
                elif choice == '0':
                    break
                else:
                    print("Invalid choice. Please try again.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"An error occurred: {e}")
        
        self.db_manager.disconnect()
    
    def _show_menu(self):
        """Display main menu"""
        print("\n" + "=" * 40)
        print("MAIN MENU")
        print("=" * 40)
        print("1. Connect to Database")
        print("2. Show Tables")
        print("3. Natural Language Query")
        print("4. Execute SQL Query")
        print("5. Create Database")
        print("6. Show Table Details")
        print("0. Exit")
    
    def _initialize_converter(self):
        """Initialize the text-to-SQL converter"""
        print("Initializing Text-to-SQL converter...")
        
        # Try OpenAI first if API key is available
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            if self.converter.initialize_openai(openai_key):
                print("✓ OpenAI GPT initialized")
                return
        
        # Fallback to local model
        if self.converter.initialize_local_model():
            print("✓ Local model initialized")
        else:
            print("⚠ Using rule-based conversion (limited functionality)")
    
    def _connect_to_database(self):
        """Handle database connection"""
        print("\n--- DATABASE CONNECTION ---")
        
        # Check available drivers
        available_types = []
        if MYSQL_AVAILABLE:
            available_types.append("mysql")
        if POSTGRESQL_AVAILABLE:
            available_types.append("postgresql")
        
        if not available_types:
            print("No database drivers available!")
            print("Install drivers with:")
            print("  pip install mysql-connector-python  # For MySQL")
            print("  pip install psycopg2-binary         # For PostgreSQL")
            return
        
        print(f"Available database types: {', '.join(available_types)}")
        
        db_type = input("Database type: ").strip().lower()
        if db_type not in available_types:
            print(f"Invalid database type. Available: {', '.join(available_types)}")
            return
        
        host = input("Host (default: localhost): ").strip() or "localhost"
        port = input(f"Port (default: {'3306' if db_type == 'mysql' else '5432'}): ").strip()
        port = int(port) if port else (3306 if db_type == 'mysql' else 5432)
        
        database = input("Database name: ").strip()
        username = input("Username: ").strip()
        password = input("Password: ").strip()
        
        config = DatabaseConfig(host, port, database, username, password, db_type)
        
        if self.db_manager.connect(config):
            print("✓ Connected successfully!")
            self._load_schema()
        else:
            print("✗ Connection failed!")
    
    def _load_schema(self):
        """Load database schema into cache"""
        print("Loading database schema...")
        self.schema_cache.clear()
        
        tables = self.db_manager.get_tables()
        for table in tables:
            table_info = self.db_manager.get_table_info(table)
            if table_info:
                self.schema_cache[table] = table_info
        
        print(f"✓ Loaded schema for {len(self.schema_cache)} tables")
    
    def _show_tables(self):
        """Display available tables"""
        if not self.db_manager.connection:
            print("Not connected to database!")
            return
        
        tables = self.db_manager.get_tables()
        if tables:
            print(f"\n--- TABLES IN DATABASE ---")
            for i, table in enumerate(tables, 1):
                print(f"{i}. {table}")
        else:
            print("No tables found in database.")
    
    def _show_table_details(self):
        """Show detailed information about a table"""
        if not self.schema_cache:
            print("No schema information available. Connect to database first.")
            return
        
        print("\n--- TABLE DETAILS ---")
        tables = list(self.schema_cache.keys())
        
        for i, table in enumerate(tables, 1):
            print(f"{i}. {table}")
        
        try:
            choice = int(input("Choose table number: ")) - 1
            if 0 <= choice < len(tables):
                table_name = tables[choice]
                table_info = self.schema_cache[table_name]
                
                print(f"\n--- {table_name.upper()} ---")
                
                # Columns
                headers = ['Column', 'Type', 'Nullable', 'Default', 'Extra']
                rows = []
                for col in table_info.columns:
                    rows.append([
                        col['name'],
                        col['type'],
                        'YES' if col['nullable'] else 'NO',
                        col['default'] or '',
                        col['extra'] or ''
                    ])
                
                print("\nColumns:")
                print(tabulate(rows, headers=headers, tablefmt='grid'))
                
                # Primary keys
                if table_info.primary_keys:
                    print(f"\nPrimary Keys: {', '.join(table_info.primary_keys)}")
                
                # Foreign keys
                if table_info.foreign_keys:
                    print("\nForeign Keys:")
                    for fk in table_info.foreign_keys:
                        print(f"  {fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}")
                
            else:
                print("Invalid table number.")
                
        except ValueError:
            print("Invalid input.")
    
    def _natural_language_query(self):
        """Handle natural language queries"""
        if not self.db_manager.connection:
            print("Not connected to database!")
            return
        
        if not self.schema_cache:
            print("No schema information available.")
            return
        
        print("\n--- NATURAL LANGUAGE QUERY ---")
        print("Examples:")
        print("  - What is the average grade of courses in 2024?")
        print("  - Count all records in the users table")
        print("  - Show me all products with price > 100")
        print()
        
        natural_query = input("Enter your question: ").strip()
        if not natural_query:
            return
        
        print("\nConverting to SQL...")
        sql_query = self.converter.convert_to_sql(natural_query, self.schema_cache)
        
        print(f"Generated SQL: {sql_query}")
        
        confirm = input("Execute this query? (y/n): ").strip().lower()
        if confirm == 'y':
            self._execute_query(sql_query)
    
    def _execute_sql_query(self):
        """Handle direct SQL query execution"""
        if not self.db_manager.connection:
            print("Not connected to database!")
            return
        
        print("\n--- EXECUTE SQL QUERY ---")
        sql_query = input("Enter SQL query: ").strip()
        
        if sql_query:
            self._execute_query(sql_query)
    
    def _execute_query(self, sql_query: str):
        """Execute SQL query and display results"""
        print("\nExecuting query...")
        success, results, message = self.db_manager.execute_query(sql_query)
        
        if success:
            print("✓ Query executed successfully!")
            if isinstance(results, list) and results:
                # Display results in table format
                if isinstance(results[0], dict):
                    headers = list(results[0].keys())
                    rows = [[row[col] for col in headers] for row in results]
                    print(f"\nResults ({len(results)} rows):")
                    print(tabulate(rows, headers=headers, tablefmt='grid'))
                else:
                    print(f"Results: {results}")
            else:
                print(f"Result: {message}")
        else:
            print(f"✗ Query failed: {message}")
    
    def _create_database(self):
        """Handle database creation"""
        if not self.db_manager.connection:
            print("Not connected to database server!")
            return
        
        print("\n--- CREATE DATABASE ---")
        db_name = input("Enter database name: ").strip()
        
        if db_name:
            if self.db_manager.create_database(db_name):
                print(f"✓ Database '{db_name}' created successfully!")
            else:
                print(f"✗ Failed to create database '{db_name}'")

def main():
    """Main entry point"""
    try:
        app = TextToSQLApp()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Application error: {e}")

if __name__ == "__main__":
    # Check dependencies
    missing_deps = []
    
    if not MYSQL_AVAILABLE:
        missing_deps.append("mysql-connector-python")
    
    if not POSTGRESQL_AVAILABLE:
        missing_deps.append("psycopg2-binary")
    
    try:
        import tabulate
    except ImportError:
        missing_deps.append("tabulate")
    
    if missing_deps:
        print("Missing dependencies. Install with:")
        for dep in missing_deps:
            print(f"  pip install {dep}")
        print()
    
    main()