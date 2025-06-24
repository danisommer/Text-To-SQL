import mysql.connector
import psycopg2
import re
import sys
import os
from dotenv import load_dotenv
from tabulate import tabulate
from typing import Dict, List, Tuple, Any, Optional, Union

# Load environment variables from .env file
load_dotenv()

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
                print(f"Tipo de banco de dados não suportado: {db_type}")
                return False
            
            print(f"Conectado ao banco de dados {db_type}: {database}")
            self.load_schema()
            return True
        except Exception as e:
            print(f"Erro ao conectar ao banco de dados: {e}")
            return False
    
    def disconnect(self) -> None:
        """Fecha a conexao com o banco de dados"""
        if self.connection:
            self.connection.close()
            print("Conexão com o banco de dados fechada")
    
    def load_schema(self) -> None:
        """Carrega o esquema do banco de dados (tabelas e suas colunas)"""
        if not self.connection:
            print("Não conectado a nenhum banco de dados")
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
            print(f"Erro ao carregar esquema: {e}")
        finally:
            cursor.close()
    
    def execute_query(self, query: str) -> Tuple[List[Tuple], List[str]]:
        """Executa consulta SQL e retorna resultados com nomes das colunas"""
        if not self.connection:
            print("Não conectado a nenhum banco de dados")
            return [], []
        
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            return results, column_names
        except Exception as e:
            print(f"Erro ao executar consulta: {e}")
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
        print("Nenhum resultado encontrado")
        return
    
    print("\n=== Resultados da Consulta ===")
    print(f"Colunas: {', '.join(column_names)}")
    print(tabulate(results, headers=column_names, tablefmt="grid"))

def print_usage_guide():
    """Imprime um guia de termos e como usar o sistema"""
    guide = """
=== Guia de Uso do Sistema de Consultas em SQL ===

Este sistema converte consultas em linguagem natural para SQL e as executa no seu banco de dados.

TIPOS DE CONSULTA SUPORTADOS:
-----------------------------------------
1. LISTAGEM BÁSICA
   Exemplos: "mostre todos os alunos", "listar disciplinas"
   SQL gerado: SELECT * FROM [tabela]

2. CONTAGEM
   Exemplos: "quantos professores existem", "contar disciplinas no departamento"
   SQL gerado: SELECT COUNT(*) FROM [tabela]

3. MÉDIAS
   Exemplos: "média de notas dos alunos", "qual a média de salário dos professores"
   SQL gerado: SELECT AVG([coluna]) FROM [tabela]

4. VALORES MÁXIMOS
   Exemplos: "maior nota da disciplina", "qual o máximo de faltas"
   SQL gerado: SELECT MAX([coluna]) FROM [tabela]

5. VALORES MÍNIMOS
   Exemplos: "menor valor de mensalidade", "mínimo de créditos"
   SQL gerado: SELECT MIN([coluna]) FROM [tabela]

CONDIÇÕES DE FILTRO:
-----------------------------------------
1. FILTRO POR ANO
   Exemplos: "disciplinas de 2024", "matrículas no ano de 2023"
   SQL gerado: WHERE YEAR(coluna_data) = [ano]

2. FILTRO DE IGUALDADE
   Exemplos: "alunos onde curso é Economia", "notas com valor igual a 10"
   SQL gerado: WHERE [coluna] = [valor]

3. OPERADORES DE COMPARAÇÃO
   Exemplos: "notas maiores que 8", "alunos com idade menor que 20" 
   SQL gerado: WHERE [coluna] > [valor], WHERE [coluna] < [valor]
   
   Operadores suportados:
   - maior que / acima de / superior a (>)
   - menor que / abaixo de / inferior a (<)
   - maior ou igual a / a partir de (>=)
   - menor ou igual a / até (<=)

DICAS PARA CONSULTAS EFETIVAS:
-----------------------------------------
- Mencione o nome da tabela claramente (ex: "alunos", "disciplinas")
- Especifique a coluna ao usar agregações (ex: "média de notas")
- Inclua condições de filtro após mencionar a solicitação principal
- Para filtrar por data, inclua o ano com uma preposição (ex: "em 2024", "do ano 2023")

OPÇÕES DE COMANDO:
-----------------------------------------
- Digite 'guia' para exibir este guia novamente
- Digite 'sair' para encerrar o programa
"""
    print(guide)

def show_tables_summary(schema: Dict[str, List[str]]) -> None:
    """Exibe um resumo compacto das tabelas disponiveis"""
    if not schema:
        print("\nNenhuma tabela disponível. Por favor, verifique sua conexão com o banco de dados.")
        return
        
    print("\n=== Tabelas e Colunas Disponíveis ===")
    table_names = list(schema.keys())
    
    for table in table_names:
        columns = schema[table]
        print(f"\n📋 {table.upper()} ({len(columns)} colunas):")
        
        if len(columns) > 6:
            col_display = ", ".join(columns[:5]) + f", ... (+{len(columns)-5} mais)"
        else:
            col_display = ", ".join(columns)
            
        print(f"   Colunas: {col_display}")

def main():
    print("=== Sistema de Consultas ao Banco de Dados ===")
    
    db = DatabaseConnection()
    
    # Solicita o tipo de banco usando números
    print("\nSelecione o tipo de banco de dados:")
    print("1 - MySQL")
    print("2 - PostgreSQL")
    
    choice = input("Opção [1]: ").strip() or "1"
    
    # Mapeia a escolha numérica para o tipo de banco
    if choice == "1":
        db_type = "mysql"
    elif choice == "2":
        db_type = "postgresql"
    else:
        print("Opção inválida. Usando MySQL como padrão.")
        db_type = "mysql"
    
    # Define automaticamente todos os parâmetros de conexão
    host = "localhost"
    
    # Solicita o nome do banco de dados
    database = input(f"\nDigite o nome do banco de dados: ").strip()
    
    # Define os parâmetros específicos para cada tipo de banco usando variáveis de ambiente
    if db_type == "mysql":
        port = int(os.getenv("MYSQL_PORT", 3306))
        user = os.getenv("MYSQL_USER", "root")
        password = os.getenv("MYSQL_PASSWORD", "")
    else:  # postgresql
        port = int(os.getenv("POSTGRESQL_PORT", 5432))
        user = os.getenv("POSTGRESQL_USER", "postgres")
        password = os.getenv("POSTGRESQL_PASSWORD", "postgres")
        
    if not db.connect(db_type, host, user, password, database, port):
        print(f"Falha ao conectar ao banco '{database}'.")
        print(f"\nVerifique se:\n- O serviço {db_type} está em execução\n- O usuário '{user}' existe\n- A senha está correta\n- O banco de dados '{database}' existe")
        return
    
    text_to_sql = TextToSQL(db)
        
    while True:
        show_tables_summary(db.schema)
        
        print("\nDigite uma consulta em linguagem natural (ou 'sair' para sair, 'guia' para ajuda):")
        query = input("> ").strip()
        
        if query.lower() in ('sair', 's', 'exit', 'quit', 'q'):
            break
        elif query.lower() in ('guia', 'guide'):
            print_usage_guide()
            continue
            
        query_info = text_to_sql.parse_query(query)
        sql = text_to_sql.generate_sql(query_info)
        
        print(f"\nSQL gerado: {sql}")
        
        if sql and not sql.startswith("Não"):
            results, column_names = db.execute_query(sql)
            display_results(results, column_names)
    
    db.disconnect()
    
if __name__ == "__main__":
    main()