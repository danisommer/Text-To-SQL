import mysql.connector
import psycopg2
import re
import sys
from tabulate import tabulate
from typing import Dict, List, Tuple, Any, Optional, Union

class DatabaseConnection:
    def __init__(self):
        self.connection = None
        self.db_type = None
        self.schema = {}
        
    def connect(self, db_type: str, host: str, user: str, password: str, database: str, port: int = None) -> bool:
        """Conecta a um banco de dados MySQL ou PostgreSQL"""
        self.db_type = db_type.lower()
        try:
            if self.db_type == 'mysql':
                self.connection = mysql.connector.connect(
                    host=host,
                    user=user,
                    password=password,
                    database=database,
                    port=port if port else 3306
                )
            elif self.db_type == 'postgresql':
                self.connection = psycopg2.connect(
                    host=host,
                    user=user,
                    password=password,
                    dbname=database,
                    port=port if port else 5432
                )
            else:
                print(f"Unsupported database type: {db_type}")
                return False
            
            print(f"Connected to {db_type} database: {database}")
            self.load_schema()
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def disconnect(self) -> None:
        """Fecha a conexao com o banco de dados"""
        if self.connection:
            self.connection.close()
            print("Database connection closed")
    
    def load_schema(self) -> None:
        """Carrega o esquema do banco de dados (tabelas e suas colunas)"""
        if not self.connection:
            print("Not connected to any database")
            return
        
        cursor = self.connection.cursor()
        
        try:
            if self.db_type == 'mysql':
                cursor.execute("SHOW TABLES")
                tables = [table[0] for table in cursor.fetchall()]
                
                for table in tables:
                    cursor.execute(f"DESCRIBE {table}")
                    columns = [col[0] for col in cursor.fetchall()]
                    self.schema[table] = columns
                    
            elif self.db_type == 'postgresql':
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                """)
                tables = [table[0] for table in cursor.fetchall()]
                
                for table in tables:
                    cursor.execute(f"""
                        SELECT column_name FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = '{table}'
                    """)
                    columns = [col[0] for col in cursor.fetchall()]
                    self.schema[table] = columns
        except Exception as e:
            print(f"Error loading schema: {e}")
        finally:
            cursor.close()
    
    def execute_query(self, query: str) -> Tuple[List[Tuple], List[str]]:
        """Executa consulta SQL e retorna resultados com nomes das colunas"""
        if not self.connection:
            print("Not connected to any database")
            return [], []
        
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            return results, column_names
        except Exception as e:
            print(f"Error executing query: {e}")
            return [], []
        finally:
            cursor.close()


class TextToSQL:
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        self.schema = db_connection.schema
        
        self.patterns = {
            'select_all': r'(?:mostre|exibir?|listar?|quais|qual|todos|todas).*(?:de|da|do|das|dos) ([\w\s]+)',
            'count': r'(?:quantos?|contar?).*(?:em|na|no|nas|nos) ([\w\s]+)',
            'average': r'(?:média|media).*(?:de|da|do|das|dos) ([\w\s]+).*(?:em|na|no|nas|nos|do|de) ([\w\s]+)',
            'max': r'(?:maior|máximo|maximo).*(?:de|da|do|das|dos) ([\w\s]+)',
            'min': r'(?:menor|mínimo|minimo).*(?:de|da|do|das|dos) ([\w\s]+)',
            'where_condition': r'(?:onde|que tenha|contendo|com)[\s]+([\w\s]+)[\s]+(?:igual a|é|eh|=|como)[\s]+([\w\s]+)',
            'greater_than': r'(?:maior(?:es)? que|acima de|superior(?:es)? a)[\s]+(\d+(?:\.\d+)?)',
            'less_than': r'(?:menor(?:es)? que|abaixo de|inferior(?:es)? a)[\s]+(\d+(?:\.\d+)?)',
            'greater_equal': r'(?:maior(?:es)? ou igual a|a partir de)[\s]+(\d+(?:\.\d+)?)',
            'less_equal': r'(?:menor(?:es)? ou igual a|até)[\s]+(\d+(?:\.\d+)?)'
        }

    def find_table_by_keyword(self, text: str) -> str:
        """Encontra a tabela mais provavel baseado em palavras-chave na consulta"""
        text = text.lower()
        best_match = None
        best_score = 0
        
        for table in self.schema.keys():
            table_lower = table.lower()
            if table_lower in text:
                score = len(table) * 2
                if score > best_score:
                    best_match = table
                    best_score = score
            
            if table_lower.endswith('s'):
                singular = table_lower[:-1]
                if singular in text:
                    score = len(singular) * 1.5
                    if score > best_score:
                        best_match = table
                        best_score = score
            else:
                plural = f"{table_lower}s"
                if plural in text:
                    score = len(table) * 1.5
                    if score > best_score:
                        best_match = table
                        best_score = score
                        
        return best_match

    def find_column_by_keyword(self, table: str, text: str) -> Optional[str]:
        """Encontra a coluna mais provavel em uma tabela baseado em palavras-chave na consulta"""
        if not table or table not in self.schema:
            return None
            
        text = text.lower()
        columns = self.schema[table]
        best_match = None
        best_score = 0
        
        for col in columns:
            col_lower = col.lower()
            if col_lower in text:
                score = len(col) * 2
                if score > best_score:
                    best_match = col
                    best_score = score
        
        return best_match
    
    def extract_year(self, text: str) -> Optional[str]:
        """Extrai o ano do texto da consulta"""
        year_pattern = r'(?:de |em |no |na |ano de |ano )(\d{4})'
        match = re.search(year_pattern, text)
        if match:
            return match.group(1)
        return None
        
    def parse_query(self, text: str) -> Dict:
        """Analisa consulta em linguagem natural e identifica componentes da consulta"""
        text = text.lower()
        query_type = 'select'
        table = self.find_table_by_keyword(text)
        
        if not table:
            return {'error': 'Não foi possível identificar a tabela na consulta'}
            
        query_info = {'type': query_type, 'table': table}
        
        if re.search(r'média|media|médias|medias', text):
            query_info['type'] = 'average'
            query_info['column'] = self.find_column_by_keyword(table, text)
        elif re.search(r'quant[oa]s|contar|conte|número|numero|total', text):
            query_info['type'] = 'count'
        elif re.search(r'máximo|maximo|maior', text) and not re.search(r'maior(?:es)? que|acima de|superior(?:es)? a', text):
            query_info['type'] = 'max'
            query_info['column'] = self.find_column_by_keyword(table, text)
        elif re.search(r'mínimo|minimo|menor', text) and not re.search(r'menor(?:es)? que|abaixo de|inferior(?:es)? a', text):
            query_info['type'] = 'min'
            query_info['column'] = self.find_column_by_keyword(table, text)
            
        where_match = re.search(self.patterns['where_condition'], text)
        if where_match:
            field_name = where_match.group(1).strip()
            value = where_match.group(2).strip()
            column = self.find_column_by_keyword(table, field_name)
            
            if column:
                query_info['where'] = {'column': column, 'value': value}
        
        greater_than_match = re.search(self.patterns['greater_than'], text)
        if greater_than_match:
            value = greater_than_match.group(1).strip()
            column = self.find_column_by_keyword(table, text) or table.lower()
            query_info.setdefault('where', {})
            query_info['where']['comparison'] = {'column': column, 'operator': '>', 'value': value}
            
        less_than_match = re.search(self.patterns['less_than'], text)
        if less_than_match:
            value = less_than_match.group(1).strip()
            column = self.find_column_by_keyword(table, text) or table.lower()
            query_info.setdefault('where', {})
            query_info['where']['comparison'] = {'column': column, 'operator': '<', 'value': value}
            
        greater_equal_match = re.search(self.patterns['greater_equal'], text)
        if greater_equal_match:
            value = greater_equal_match.group(1).strip()
            column = self.find_column_by_keyword(table, text) or table.lower()
            query_info.setdefault('where', {})
            query_info['where']['comparison'] = {'column': column, 'operator': '>=', 'value': value}
            
        less_equal_match = re.search(self.patterns['less_equal'], text)
        if less_equal_match:
            value = less_equal_match.group(1).strip()
            column = self.find_column_by_keyword(table, text) or table.lower()
            query_info.setdefault('where', {})
            query_info['where']['comparison'] = {'column': column, 'operator': '<=', 'value': value}
                
        year = self.extract_year(text)
        if year:
            date_columns = [col for col in self.schema[table] 
                           if any(date_term in col.lower() for date_term in 
                                 ['data', 'date', 'dt', 'ano', 'year'])]
            
            if date_columns:
                query_info.setdefault('where', {})
                query_info['where']['year'] = {'column': date_columns[0], 'value': year}
                
        return query_info
        
    def generate_sql(self, query_info: Dict) -> str:
        """Gera SQL a partir das informacoes da consulta analisada"""
        if 'error' in query_info:
            return query_info['error']
        
        if query_info.get('type') == 'columns':
            table = query_info['table']
            if self.db.db_type == 'mysql':
                return f"SHOW COLUMNS FROM {table}"
            else:
                return f"SELECT column_name, data_type, character_maximum_length FROM information_schema.columns WHERE table_name = '{table}'"
        
        table = query_info['table']
        query_type = query_info['type']
        sql = ""
        
        if query_type == 'select':
            sql = f"SELECT * FROM {table}"
        elif query_type == 'count':
            sql = f"SELECT COUNT(*) FROM {table}"
        elif query_type == 'average':
            column = query_info.get('column', '*')
            sql = f"SELECT AVG({column}) FROM {table}"
        elif query_type == 'max':
            column = query_info.get('column', '*')
            sql = f"SELECT MAX({column}) FROM {table}"
        elif query_type == 'min':
            column = query_info.get('column', '*')
            sql = f"SELECT MIN({column}) FROM {table}"
            
        if 'where' in query_info:
            conditions = []
            
            if 'column' in query_info['where']:
                column = query_info['where']['column']
                value = query_info['where']['value']
                if value.isdigit():
                    conditions.append(f"{column} = {value}")
                else:
                    conditions.append(f"{column} = '{value}'")
            
            if 'comparison' in query_info['where']:
                comp_info = query_info['where']['comparison']
                column = comp_info['column']
                operator = comp_info['operator']
                value = comp_info['value']
                conditions.append(f"{column} {operator} {value}")
                    
            if 'year' in query_info['where']:
                year_col = query_info['where']['year']['column']
                year_val = query_info['where']['year']['value']
                
                if self.db.db_type == 'mysql':
                    conditions.append(f"YEAR({year_col}) = {year_val}")
                else:
                    conditions.append(f"EXTRACT(YEAR FROM {year_col}) = {year_val}")
                    
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
                
        return sql


def display_results(results: List[Tuple], column_names: List[str]) -> None:
    """Exibe resultados da consulta em uma tabela formatada"""
    if not results:
        print("No results found")
        return
    
    print("\n=== Query Results ===")
    print(f"Columns: {', '.join(column_names)}")
    print(tabulate(results, headers=column_names, tablefmt="grid"))

def print_usage_guide():
    """Imprime um guia de termos e como usar o sistema"""
    guide = """
=== Text-to-SQL Query System Usage Guide ===

This system converts natural language queries to SQL and executes them against your database.

SUPPORTED QUERY TYPES:
-----------------------------------------
1. BASIC LISTING
   Examples: "mostre todos os alunos", "listar disciplinas"
   SQL generated: SELECT * FROM [table]

2. COUNTING
   Examples: "quantos professores existem", "contar disciplinas no departamento"
   SQL generated: SELECT COUNT(*) FROM [table]

3. AVERAGES
   Examples: "média de notas dos alunos", "qual a média de salário dos professores"
   SQL generated: SELECT AVG([column]) FROM [table]

4. MAXIMUM VALUES
   Examples: "maior nota da disciplina", "qual o máximo de faltas"
   SQL generated: SELECT MAX([column]) FROM [table]

5. MINIMUM VALUES
   Examples: "menor valor de mensalidade", "mínimo de créditos"
   SQL generated: SELECT MIN([column]) FROM [table]

FILTERING CONDITIONS:
-----------------------------------------
1. YEAR FILTER
   Examples: "disciplinas de 2024", "matrículas no ano de 2023"
   SQL generated: WHERE YEAR(date_column) = [year]

2. EQUALITY FILTER
   Examples: "alunos onde curso é Economia", "notas com valor igual a 10"
   SQL generated: WHERE [column] = [value]

3. COMPARISON OPERATORS
   Examples: "notas maiores que 8", "alunos com idade menor que 20" 
   SQL generated: WHERE [column] > [value], WHERE [column] < [value]
   
   Supported operators:
   - maior que / acima de / superior a (>)
   - menor que / abaixo de / inferior a (<)
   - maior ou igual a / a partir de (>=)
   - menor ou igual a / até (<=)

TIPS FOR EFFECTIVE QUERIES:
-----------------------------------------
- Mention the table name clearly (e.g., "alunos", "disciplinas")
- Specify the column when using aggregations (e.g., "média de notas")
- Include filtering conditions after mentioning the main request
- For date filtering, include the year with a preposition (e.g., "em 2024", "do ano 2023")

COMMAND OPTIONS:
-----------------------------------------
- Type 'guide' to display this guide again
- Type 'tables' to show available tables and columns
- Type 'exit' to quit the program
"""
    print(guide)

def display_tables(schema):
    """Exibe todas as tabelas e suas colunas"""
    print("\n=== Available Tables and Columns ===")
    for table, columns in schema.items():
        print(f"\n📋 {table.upper()}")
        for col in columns:
            print(f"  - {col}")

def show_tables_summary(schema: Dict[str, List[str]]) -> None:
    """Exibe um resumo compacto das tabelas disponiveis"""
    if not schema:
        print("\nNo tables available. Please check your database connection.")
        return
        
    print("\n=== Available Tables and Columns ===")
    table_names = list(schema.keys())
    
    for table in table_names:
        columns = schema[table]
        print(f"\n📋 {table.upper()} ({len(columns)} columns):")
        
        if len(columns) > 6:
            col_display = ", ".join(columns[:5]) + f", ... (+{len(columns)-5} more)"
        else:
            col_display = ", ".join(columns)
            
        print(f"   Columns: {col_display}")
    
    print("\nType 'tables' for more detailed column information")

def main():
    print("=== Database Query System ===")
    
    db = DatabaseConnection()
    
    db_type = input("Database type (mysql/postgresql): ").strip().lower()
    host = input("Host [localhost]: ").strip() or "localhost"
    port = input("Port [default]: ").strip()
    port = int(port) if port.isdigit() else None
    user = input("Username: ").strip()
    password = input("Password: ").strip()
    database = input("Database name: ").strip()
    
    if not db.connect(db_type, host, user, password, database, port):
        print("Failed to connect to database")
        return
    
    text_to_sql = TextToSQL(db)
    
    print("\nAvailable tables:")
    for table, columns in db.schema.items():
        print(f"- {table}: {', '.join(columns)}")
    
    print("\nType 'guide' to see usage instructions and supported query types.")
    
    while True:
        show_tables_summary(db.schema)
        
        print("\nEnter a natural language query (or 'exit' to quit, 'guide' for help, 'tables' to list tables):")
        query = input("> ").strip()
        
        if query.lower() in ('exit', 'quit', 'q'):
            break
        elif query.lower() == 'guide':
            print_usage_guide()
            continue
        elif query.lower() == 'tables':
            display_tables(db.schema)
            continue
            
        query_info = text_to_sql.parse_query(query)
        sql = text_to_sql.generate_sql(query_info)
        
        print(f"\nGenerated SQL: {sql}")
        
        if sql and not sql.startswith("Não"):
            results, column_names = db.execute_query(sql)
            display_results(results, column_names)
    
    db.disconnect()
    
if __name__ == "__main__":
    main()