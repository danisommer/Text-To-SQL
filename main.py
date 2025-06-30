import mysql.connector
import psycopg2
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import logging as transformers_logging
from dotenv import load_dotenv
from tabulate import tabulate
from typing import Dict, List, Tuple, Any, Optional, Union

# Carrega variáveis de ambiente e configura o registro
load_dotenv()
transformers_logging.set_verbosity_error()

class DatabaseConnection:
    def __init__(self):
        self.connection = None
        self.db_type = None
        self.schema = {}
        self.column_types = {}
        self.foreign_keys = {}
        self.primary_keys = {}
        self.table_comments = {}
        
    def connect(self, db_type: str, host: str, user: str, password: str, database: str, port: int = None) -> bool:
        """Conecta a um banco de dados MySQL ou PostgreSQL"""
        self.db_type = db_type.lower()
        try:
            if self.db_type == 'mysql':
                self.connection = mysql.connector.connect(
                    host=host, user=user, password=password, database=database, port=port if port else 3306
                )
            elif self.db_type == 'postgresql':
                self.connection = psycopg2.connect(
                    host=host, user=user, password=password, dbname=database, port=port if port else 5432
                )
            else:
                print(f"Tipo de banco de dados não suportado: {db_type}")
                return False
            
            print(f"Conectado ao banco {database}")
            self.load_schema()
            return True
        except Exception as e:
            print(f"Erro ao conectar: {e}")
            return False
    
    def disconnect(self) -> None:
        """Fecha a conexao com o banco de dados"""
        if self.connection:
            self.connection.close()
            print("Conexão fechada")
    
    def load_schema(self) -> None:
        """Carrega o esquema do banco de dados"""
        if not self.connection:
            print("Não conectado a nenhum banco de dados")
            return
        
        cursor = self.connection.cursor()
        
        try:
            if self.db_type == 'mysql':
                # Carrega tabelas e colunas
                cursor.execute("SHOW TABLES")
                tables = [table[0] for table in cursor.fetchall()]
                
                for table in tables:
                    # Obtém colunas
                    cursor.execute(f"DESCRIBE {table}")
                    table_info = cursor.fetchall()
                    columns = [col[0] for col in table_info]
                    column_types = {col[0]: col[1] for col in table_info}
                    self.schema[table] = columns
                    self.column_types[table] = column_types
                    
                    # Obtém chaves primárias
                    cursor.execute(f"SHOW KEYS FROM {table} WHERE Key_name = 'PRIMARY'")
                    pk_info = cursor.fetchall()
                    self.primary_keys[table] = [pk[4] for pk in pk_info]
                    
                    # Obtém chaves estrangeiras
                    cursor.execute(f"""
                        SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                        WHERE TABLE_NAME = '{table}' AND REFERENCED_TABLE_NAME IS NOT NULL
                    """)
                    fk_info = cursor.fetchall()
                    self.foreign_keys[table] = [(fk[0], fk[1], fk[2]) for fk in fk_info]
                    
                    # Obtém comentários da tabela
                    cursor.execute(f"""
                        SELECT TABLE_COMMENT FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table}'
                    """)
                    comment_info = cursor.fetchone()
                    self.table_comments[table] = comment_info[0] if comment_info and comment_info[0] else ""
                    
            elif self.db_type == 'postgresql':
                # Carrega tabelas
                cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
                tables = [table[0] for table in cursor.fetchall()]
                
                for table in tables:
                    # Obtém colunas
                    cursor.execute(f"""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = '{table}'
                        ORDER BY ordinal_position
                    """)
                    table_info = cursor.fetchall()
                    columns = [col[0] for col in table_info]
                    column_types = {col[0]: f"{col[1]} {'NULL' if col[2]=='YES' else 'NOT NULL'}" for col in table_info}
                    self.schema[table] = columns
                    self.column_types[table] = column_types
                    
                    # Obtém chaves primárias
                    cursor.execute(f"""
                        SELECT column_name FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
                        WHERE tc.table_name = '{table}' AND tc.constraint_type = 'PRIMARY KEY'
                    """)
                    pk_info = cursor.fetchall()
                    self.primary_keys[table] = [pk[0] for pk in pk_info]
                    
                    # Obtém chaves estrangeiras
                    cursor.execute(f"""
                        SELECT kcu.column_name, ccu.table_name, ccu.column_name
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
                        JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name
                        WHERE tc.table_name = '{table}' AND tc.constraint_type = 'FOREIGN KEY'
                    """)
                    fk_info = cursor.fetchall()
                    self.foreign_keys[table] = [(fk[0], fk[1], fk[2]) for fk in fk_info]
                    
                    # Obtém comentários da tabela
                    cursor.execute(f"""
                        SELECT obj_description(oid) FROM pg_class WHERE relname = '{table}' AND relkind = 'r'
                    """)
                    comment_info = cursor.fetchone()
                    self.table_comments[table] = comment_info[0] if comment_info and comment_info[0] else ""
                    
        except Exception as e:
            print(f"Erro ao carregar esquema: {e}")
        finally:
            cursor.close()
    
    def execute_query(self, query: str) -> Tuple[List[Tuple], List[str]]:
        """Executa consulta SQL e retorna resultados com nomes das colunas"""
        if not self.connection:
            print("Não conectado ao banco")
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
        self.column_types = db_connection.column_types
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Carrega o modelo Llama 3.2"""
        try:
            print("Carregando modelo Llama 3.2...")
            
            model_name = os.getenv("LLAMA_MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
            use_cpu_only = os.getenv("USE_CPU_ONLY", "false").lower() == "true"
            device_map = "cpu" if use_cpu_only or not torch.cuda.is_available() else "auto"
            torch_dtype = torch.float32 if use_cpu_only or not torch.cuda.is_available() else torch.float16
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            
            # Cria pipeline para geração de texto
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            print("Modelo carregado com sucesso")
            self.model_loaded = True
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            raise Exception(f"Falha ao carregar modelo: {e}")

    def format_schema_for_prompt(self):
        """Formata o esquema do banco de dados para o prompt"""
        schema_text = "DATABASE SCHEMA:\n\n"
        
        # Adiciona visão geral das tabelas
        schema_text += "TABLES:\n"
        for table_name in self.schema.keys():
            schema_text += f"- {table_name}\n"
        schema_text += "\n"
        
        # Adiciona detalhes para cada tabela
        for table_name, columns in self.schema.items():
            schema_text += f"TABLE: {table_name}\n"
            
            # Chaves primárias
            primary_keys = self.db.primary_keys.get(table_name, [])
            if primary_keys:
                schema_text += f"Primary Key(s): {', '.join(primary_keys)}\n"
            
            # Colunas com tipos
            schema_text += "Columns:\n"
            for column in columns:
                col_type = self.column_types.get(table_name, {}).get(column, "unknown")
                schema_text += f"  - {column}: {col_type}\n"
            
            # Chaves estrangeiras
            foreign_keys = self.db.foreign_keys.get(table_name, [])
            if foreign_keys:
                schema_text += "Foreign Keys:\n"
                for fk_col, ref_table, ref_col in foreign_keys:
                    schema_text += f"  - {fk_col} -> {ref_table}.{ref_col}\n"
            
            schema_text += "\n"
        
        return schema_text
    
    def generate_sql_with_llama(self, text: str) -> str:
        """Gera SQL usando o modelo Llama"""
        if not self.model_loaded or self.pipe is None:
            raise Exception("Modelo Llama não carregado")
        
        schema_text = self.format_schema_for_prompt()
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert SQL assistant. Convert Portuguese natural language queries to SQL using the provided database schema.

{schema_text}

GUIDELINES:
- Respond with ONLY the SQL query, no explanations
- Use proper JOIN syntax when accessing multiple tables
- Use the exact table and column names from the schema
- Handle Portuguese text and accents correctly

<|eot_id|><|start_header_id|>user<|end_header_id|>

Convert this Portuguese query to SQL: {text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        response = self.pipe(
            prompt,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        if response and len(response) > 0:
            generated_text = response[0]["generated_text"]
            return self.extract_sql_from_response(generated_text)
        else:
            raise Exception("Resposta vazia do modelo")

    def extract_sql_from_response(self, response: str) -> str:
        """Extrai a consulta SQL da resposta do modelo"""
        # Remove qualquer formatação markdown e extrai o SQL
        sql_match = re.search(r"SQL query:\s*(.*?)(?:$|```)", response, re.DOTALL)
        
        if sql_match:
            sql = sql_match.group(1).strip()
            sql = re.sub(r"```sql|```", "", sql).strip()
            return sql
        
        # Alternativa: encontra qualquer parte que pareça SQL
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if 'SELECT' in line.upper() or 'WITH' in line.upper():
                sql_lines = []
                for j in range(i, len(lines)):
                    if lines[j].strip() and not lines[j].startswith('```'):
                        sql_lines.append(lines[j].strip())
                    if j > i and (lines[j].startswith('```') or lines[j].startswith('Note:') or lines[j].startswith('Explanation:')):
                        break
                return ' '.join(sql_lines)
        
        # Alternativa: retorna tudo depois de "SQL query:"
        if "SQL query:" in response:
            return response.split("SQL query:")[1].strip()
        
        return response.strip()
    
    def parse_query(self, text: str) -> Dict:
        """Usa Llama para analisar a consulta em linguagem natural"""
        try:
            sql = self.generate_sql_with_llama(text)
            return {
                "raw_sql": sql,
                "parsed": True
            }
        except Exception as e:
            return {
                "error": f"Erro na geração do SQL: {e}",
                "parsed": False
            }
    
    def generate_sql(self, query_info: Dict) -> str:
        """Gera SQL a partir das informações da consulta"""
        if "error" in query_info:
            return query_info["error"]
        
        return query_info["raw_sql"]

def display_results(results: List[Tuple], column_names: List[str]) -> None:
    """Exibe resultados com formatação melhorada"""
    if not results:
        print("Nenhum resultado encontrado")
        return
    
    print(f"\nEncontrados {len(results)} resultado(s)")
    print("=" * 50)
    
    if len(results) <= 20:
        print(tabulate(results, headers=column_names, tablefmt="grid"))
    else:
        print(f"Mostrando primeiros 20 de {len(results)} resultados:")
        print(tabulate(results[:20], headers=column_names, tablefmt="grid"))
        print(f"\n... e mais {len(results) - 20} resultado(s)")

def print_usage_guide():
    """Guia de uso do sistema"""
    guide = """
SISTEMA TEXT-TO-SQL COM LLAMA 3.2

EXEMPLOS DE CONSULTAS:
   "mostre todos os alunos"
   "quantos professores existem"
   "média de notas dos alunos"
   "maior nota registrada"
   "alunos com notas maiores que 8"
   "professores de matemática"

COMANDOS:
   'guia' - Mostra este guia
   'sair' - Encerra o programa
"""
    print(guide)

def show_tables_summary(schema: Dict[str, List[str]], column_types: Dict[str, Dict[str, str]], 
                       primary_keys: Dict[str, List[str]] = None, 
                       foreign_keys: Dict[str, List[Tuple[str, str, str]]] = None) -> None:
    """Exibe resumo melhorado das tabelas com relacionamentos"""
    if not schema:
        print("\nNenhuma tabela disponível")
        return
        
    print("\nESTRUTURA DO BANCO:")
    
    for table in schema.keys():
        columns = schema[table]
        types = column_types.get(table, {})
        pks = primary_keys.get(table, []) if primary_keys else []
        fks = foreign_keys.get(table, []) if foreign_keys else []
        
        print(f"\n{table.upper()} ({len(columns)} colunas)")
        print("-" * 40)
        
        # Mostra chaves primárias
        if pks:
            print(f"   🔑 Chave(s) primária(s): {', '.join(pks)}")
        
        # Mostra colunas
        for col in columns[:6]:  # Mostra até 6 colunas
            col_type = types.get(col, 'unknown')
            pk_marker = " [PK]" if col in pks else ""
            fk_marker = ""
            for fk_col, ref_table, ref_col in fks:
                if fk_col == col:
                    fk_marker = f" -> {ref_table}.{ref_col}"
                    break
            print(f"   • {col} ({col_type}){pk_marker}{fk_marker}")
        
        if len(columns) > 6:
            print(f"   ... e mais {len(columns) - 6} coluna(s)")
        
        # Mostra relacionamentos de chave estrangeira
        if fks:
            print("   🔗 Relacionamentos:")
            for fk_col, ref_table, ref_col in fks[:3]:  # Mostra até 3 FKs
                print(f"     {fk_col} -> {ref_table}.{ref_col}")

def main():
    print("SISTEMA TEXT-TO-SQL COM LLAMA 3.2")
    
    db = DatabaseConnection()
    
    print("\nSelecione o tipo de banco:")
    print("1. MySQL")
    print("2. PostgreSQL")
    
    choice = input("\nSua escolha [1]: ").strip() or "1"
    
    db_type = "mysql" if choice == "1" else "postgresql" if choice == "2" else "mysql"
    
    if choice not in ["1", "2"]:
        print("Opção inválida. Usando MySQL.")
        db_type = "mysql"
    
    host = "localhost"
    database = input(f"\nNome do banco: ").strip()
    
    if db_type == "mysql":
        port = int(os.getenv("MYSQL_PORT", 3306))
        user = os.getenv("MYSQL_USER", "root")
        password = os.getenv("MYSQL_PASSWORD", "")
    else:
        port = int(os.getenv("POSTGRESQL_PORT", 5432))
        user = os.getenv("POSTGRESQL_USER", "postgres")
        password = os.getenv("POSTGRESQL_PASSWORD", "postgres")
        
    print(f"\nConectando ao {db_type.upper()} em {host}:{port}...")
    
    if not db.connect(db_type, host, user, password, database, port):
        print(f"\nFalha na conexão com '{database}'")
        return
    
    try:
        print("\nIniciando modelo Llama 3.2 1B...")
        text_to_sql = TextToSQL(db)
        print("\nSistema pronto! Digite suas consultas em português.")
    except Exception as e:
        print(f"\nErro ao inicializar modelo: {e}")
        return
    
    while True:
        show_tables_summary(db.schema, db.column_types, db.primary_keys, db.foreign_keys)
        
        print(f"\nDigite sua consulta (ou 'sair' para sair, 'guia' para ajuda):")
        query = input("> ").strip()
        
        if query.lower() in ('sair', 's', 'exit', 'quit', 'q'):
            print("Até logo!")
            break
        elif query.lower() in ('guia', 'guide', 'help', 'ajuda'):
            print_usage_guide()
            continue
        elif not query:
            continue
            
        print(f"\nProcessando: '{query}'")
        
        try:
            query_info = text_to_sql.parse_query(query)
            sql = text_to_sql.generate_sql(query_info)
            
            print(f"\nSQL gerado: {sql}")
            
            results, column_names = db.execute_query(sql)
            display_results(results, column_names)
        except Exception as e:
            print(f"Erro: {e}")
    
    db.disconnect()

if __name__ == "__main__":
    main()