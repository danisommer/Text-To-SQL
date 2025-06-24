import mysql.connector
import psycopg2
import re
import sys
import os
from dotenv import load_dotenv
from tabulate import tabulate
from typing import Dict, List, Tuple, Any, Optional, Union
from difflib import SequenceMatcher

# Load environment variables from .env file
load_dotenv()

class DatabaseConnection:
    def __init__(self):
        self.connection = None
        self.db_type = None
        self.schema = {}
        self.column_types = {}
        
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
        """Carrega o esquema do banco de dados com tipos de dados"""
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
                    table_info = cursor.fetchall()
                    columns = [col[0] for col in table_info]
                    column_types = {col[0]: col[1] for col in table_info}
                    self.schema[table] = columns
                    self.column_types[table] = column_types
                    
            elif self.db_type == 'postgresql':
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                """)
                tables = [table[0] for table in cursor.fetchall()]
                
                for table in tables:
                    cursor.execute(f"""
                        SELECT column_name, data_type FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = '{table}'
                        ORDER BY ordinal_position
                    """)
                    table_info = cursor.fetchall()
                    columns = [col[0] for col in table_info]
                    column_types = {col[0]: col[1] for col in table_info}
                    self.schema[table] = columns
                    self.column_types[table] = column_types
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
        self.column_types = db_connection.column_types
        
        # Enhanced patterns with better Portuguese support
        self.patterns = {
            'select_all': r'(?:mostre|exibir?|listar?|quais|qual|todos|todas|ver|visualizar).*?(?:de|da|do|das|dos|na|no|nas|nos)\s+([\w\s]+)',
            'count': r'(?:quantos?|quantas?|contar?|conte|número|numero|total).*?(?:de|da|do|das|dos|em|na|no|nas|nos)\s+([\w\s]+)',
            'average': r'(?:média|media|médias|medias).*?(?:de|da|do|das|dos)\s+([\w\s]+).*?(?:em|na|no|nas|nos|do|de)\s+([\w\s]+)',
            'max': r'(?:maior|máximo|maximo|máxima|maxima).*?(?:de|da|do|das|dos)\s+([\w\s]+)',
            'min': r'(?:menor|mínimo|minimo|mínima|minima).*?(?:de|da|do|das|dos)\s+([\w\s]+)',
            'where_condition': r'(?:onde|que tenha|contendo|com)\s+([\w\s]+)\s+(?:igual a|é|eh|=|como|seja)\s+([\w\s\'\"]+)',
            'greater_than': r'(?:maior(?:es)? que|acima de|superior(?:es)? a)\s+(\d+(?:\.\d+)?)',
            'less_than': r'(?:menor(?:es)? que|abaixo de|inferior(?:es)? a)\s+(\d+(?:\.\d+)?)',
            'greater_equal': r'(?:maior(?:es)? ou igual a|a partir de)\s+(\d+(?:\.\d+)?)',
            'less_equal': r'(?:menor(?:es)? ou igual a|até)\s+(\d+(?:\.\d+)?)',
            'join_indicators': r'(?:junto com|com|relacionado|vinculado|associado)',
            'group_by': r'(?:agrupar|agrupado|por cada|para cada)\s+([\w\s]+)',
            'order_by': r'(?:ordenar|ordenado|classificar)\s+(?:por\s+)?([\w\s]+)(?:\s+(crescente|decrescente|asc|desc))?'
        }

    def similarity(self, a: str, b: str) -> float:
        """Calcula similaridade entre duas strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def find_table_by_keyword(self, text: str) -> str:
        """Encontra a tabela mais provavel com melhor matching"""
        text = text.lower()
        best_match = None
        best_score = 0
        
        # Mapeamento de palavras comuns para tabelas
        table_aliases = {
            'aluno': 'alunos',
            'estudante': 'alunos',
            'estudantes': 'alunos',
            'professor': 'professores',
            'docente': 'professores',
            'docentes': 'professores',
            'materia': 'materias',
            'disciplina': 'materias',
            'disciplinas': 'materias',
            'nota': 'notas',
            'avaliacao': 'notas',
            'avaliacoes': 'notas',
            'avaliação': 'notas',
            'avaliações': 'notas'
        }
        
        # Primeiro, verifica aliases
        for alias, table in table_aliases.items():
            if alias in text and table in self.schema:
                return table
        
        # Depois verifica similaridade com nomes de tabelas
        for table in self.schema.keys():
            similarity_score = self.similarity(table, text)
            if similarity_score > 0.6 and similarity_score > best_score:
                best_match = table
                best_score = similarity_score
            
            # Verifica se a tabela está presente no texto
            if table.lower() in text:
                score = len(table) * 2
                if score > best_score:
                    best_match = table
                    best_score = score
                        
        return best_match

    def find_column_by_keyword(self, table: str, text: str, context: str = '') -> Optional[str]:
        """Encontra a coluna mais provavel com contexto melhorado"""
        if not table or table not in self.schema:
            return None
            
        text = text.lower()
        columns = self.schema[table]
        best_match = None
        best_score = 0
        
        # Mapeamento de palavras para colunas comuns
        column_aliases = {
            'nome': 'nome',
            'nomes': 'nome',
            'data': 'data_nascimento' if 'nascimento' in text else 'data_avaliacao',
            'nascimento': 'data_nascimento',
            'idade': 'data_nascimento',
            'especialidade': 'especialidade',
            'area': 'especialidade',
            'nota': 'nota',
            'notas': 'nota',
            'pontuacao': 'nota',
            'pontuações': 'nota',
            'avaliacao': 'data_avaliacao',
            'avaliação': 'data_avaliacao'
        }
        
        # Verifica aliases primeiro
        for alias, col_name in column_aliases.items():
            if alias in text and col_name in columns:
                return col_name
        
        # Verifica similaridade
        for col in columns:
            if col.lower() in text:
                score = len(col) * 2
                if score > best_score:
                    best_match = col
                    best_score = score
            
            similarity_score = self.similarity(col, text)
            if similarity_score > 0.7 and similarity_score > best_score:
                best_match = col
                best_score = similarity_score
        
        return best_match
    
    def extract_year(self, text: str) -> Optional[str]:
        """Extrai o ano do texto da consulta com mais padrões"""
        year_patterns = [
            r'(?:de |em |no |na |ano de |ano )(\d{4})',
            r'(\d{4})',
            r'(?:desde |a partir de )(\d{4})',
            r'(?:até |ate )(\d{4})'
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text)
            if match:
                year = int(match.group(1))
                if 1900 <= year <= 2100:  # Validação básica de ano
                    return str(year)
        return None
    
    def detect_joins(self, text: str) -> List[Dict]:
        """Detecta necessidade de JOINs baseado no contexto"""
        joins = []
        text = text.lower()
        
        # Mapeamento de relacionamentos conhecidos
        relationships = {
            ('alunos', 'notas'): ('id', 'id_aluno'),
            ('materias', 'notas'): ('id', 'id_materia'),
            ('professores', 'materias'): ('id', 'id_professor')
        }
        
        # Detecta se múltiplas tabelas são mencionadas
        mentioned_tables = []
        for table in self.schema.keys():
            if table in text or any(alias in text for alias in ['aluno', 'professor', 'materia', 'nota']):
                if table == 'alunos' and ('aluno' in text or 'estudante' in text):
                    mentioned_tables.append(table)
                elif table == 'professores' and ('professor' in text or 'docente' in text):
                    mentioned_tables.append(table)
                elif table == 'materias' and ('materia' in text or 'disciplina' in text):
                    mentioned_tables.append(table)
                elif table == 'notas' and ('nota' in text or 'avaliacao' in text):
                    mentioned_tables.append(table)
        
        # Gera JOINs necessários
        for i, table1 in enumerate(mentioned_tables):
            for table2 in mentioned_tables[i+1:]:
                for (t1, t2), (col1, col2) in relationships.items():
                    if (table1 == t1 and table2 == t2) or (table1 == t2 and table2 == t1):
                        joins.append({
                            'table1': t1,
                            'table2': t2,
                            'column1': col1,
                            'column2': col2
                        })
        
        return joins
        
    def parse_multiple_conditions(self, text: str, table: str) -> List[Dict]:
        """Analisa múltiplas condições WHERE conectadas por 'e'/'and'"""
        conditions = []
        
        # Divide o texto por conectores
        parts = re.split(r'\s+e\s+|\s+and\s+', text)
        
        for part in parts:
            part = part.strip()
            
            # Condição de igualdade
            where_match = re.search(r'([\w\s]+)\s+(?:é|eh|=|igual a|seja)\s+([\w\s\'\"]+)', part)
            if where_match:
                field_name = where_match.group(1).strip()
                value = where_match.group(2).strip().strip('\'"')
                column = self.find_column_by_keyword(table, field_name)
                if column:
                    conditions.append({'type': 'equality', 'column': column, 'value': value})
                continue
            
            # Filtro por ano específico
            year_match = re.search(r'ano\s+(?:de\s+)?(?:nascimento\s+)?(?:é|eh|=)\s+(\d{4})', part)
            if year_match:
                year = year_match.group(1)
                date_columns = [col for col in self.schema[table] 
                               if any(date_term in col.lower() for date_term in 
                                     ['data', 'date', 'dt', 'ano', 'year'])]
                if date_columns:
                    conditions.append({'type': 'year', 'column': date_columns[0], 'value': year})
                continue
            
            # Operadores de comparação
            for op_name, pattern in [
                ('greater_than', r'(?:maior(?:es)? que|acima de|superior(?:es)? a)\s+(\d+(?:\.\d+)?)'),
                ('less_than', r'(?:menor(?:es)? que|abaixo de|inferior(?:es)? a)\s+(\d+(?:\.\d+)?)'),
                ('greater_equal', r'(?:maior(?:es)? ou igual a|a partir de)\s+(\d+(?:\.\d+)?)'),
                ('less_equal', r'(?:menor(?:es)? ou igual a|até)\s+(\d+(?:\.\d+)?)')
            ]:
                match = re.search(pattern, part)
                if match:
                    value = match.group(1).strip()
                    column = self.find_column_by_keyword(table, part) or 'nota'
                    operator_map = {
                        'greater_than': '>',
                        'less_than': '<',
                        'greater_equal': '>=',
                        'less_equal': '<='
                    }
                    conditions.append({
                        'type': 'comparison',
                        'column': column, 
                        'operator': operator_map[op_name], 
                        'value': value
                    })
                    break
        
        return conditions

    def parse_query(self, text: str) -> Dict:
        """Analisa consulta com detecção melhorada"""
        original_text = text
        text = text.lower()
        query_type = 'select'
        table = self.find_table_by_keyword(text)
        
        if not table:
            return {'error': 'Não foi possível identificar a tabela na consulta'}
            
        query_info = {'type': query_type, 'table': table}
        
        # Detecta tipo de operação com melhor busca por colunas
        if re.search(r'média|media|médias|medias', text):
            query_info['type'] = 'average'
            # Busca coluna após "média de" ou similar
            avg_match = re.search(r'média\s+(?:de\s+|da\s+|das\s+|do\s+|dos\s+)?([\w\s]+)', text)
            if avg_match:
                column_hint = avg_match.group(1).strip()
                query_info['column'] = self.find_column_by_keyword(table, column_hint, 'average')
            if not query_info.get('column'):
                query_info['column'] = 'nota'  # Default
                
        elif re.search(r'quant[oa]s|contar|conte|número|numero|total', text):
            query_info['type'] = 'count'
            
        elif re.search(r'máximo|maximo|maior', text) and not re.search(r'maior(?:es)? que|acima de|superior(?:es)? a', text):
            query_info['type'] = 'max'
            # Busca coluna após "maior" ou "máximo"
            max_match = re.search(r'(?:maior|máximo|maximo)\s+(?:de\s+|da\s+|das\s+|do\s+|dos\s+)?([\w\s]+)', text)
            if max_match:
                column_hint = max_match.group(1).strip()
                query_info['column'] = self.find_column_by_keyword(table, column_hint, 'max')
            if not query_info.get('column'):
                # Se não encontrou coluna específica, tenta detectar contexto
                if 'nota' in text or 'avaliacao' in text or 'pontuacao' in text:
                    query_info['column'] = 'nota'
                else:
                    query_info['column'] = 'nota'  # Default
                    
        elif re.search(r'mínimo|minimo|menor', text) and not re.search(r'menor(?:es)? que|abaixo de|inferior(?:es)? a', text):
            query_info['type'] = 'min'
            # Busca coluna após "menor" ou "mínimo"
            min_match = re.search(r'(?:menor|mínimo|minimo)\s+(?:de\s+|da\s+|das\s+|do\s+|dos\s+)?([\w\s]+)', text)
            if min_match:
                column_hint = min_match.group(1).strip()
                query_info['column'] = self.find_column_by_keyword(table, column_hint, 'min')
            if not query_info.get('column'):
                query_info['column'] = 'nota'  # Default
        
        # Detecta JOINs necessários
        joins = self.detect_joins(text)
        if joins:
            query_info['joins'] = joins
            
        # Detecta condições WHERE com suporte a múltiplas condições
        where_part = ""
        if ' onde ' in text:
            where_part = text.split(' onde ', 1)[1]
        elif ' que ' in text and ('tenha' in text or 'tem' in text):
            where_part = text.split(' que ', 1)[1]
        elif ' com ' in text:
            where_part = text.split(' com ', 1)[1]
            
        if where_part:
            conditions = self.parse_multiple_conditions(where_part, table)
            if conditions:
                query_info['where'] = {'conditions': conditions}
        
        # Se não encontrou condições estruturadas, tenta padrões simples
        if 'where' not in query_info:
            # Condição simples de igualdade
            where_match = re.search(self.patterns['where_condition'], text)
            if where_match:
                field_name = where_match.group(1).strip()
                value = where_match.group(2).strip().strip('\'"')
                column = self.find_column_by_keyword(table, field_name)
                
                if column:
                    query_info['where'] = {'conditions': [{'type': 'equality', 'column': column, 'value': value}]}
            
            # Operadores de comparação únicos
            for op_name, pattern in [
                ('greater_than', self.patterns['greater_than']),
                ('less_than', self.patterns['less_than']),
                ('greater_equal', self.patterns['greater_equal']),
                ('less_equal', self.patterns['less_equal'])
            ]:
                match = re.search(pattern, text)
                if match:
                    value = match.group(1).strip()
                    column = self.find_column_by_keyword(table, text) or 'nota'
                    operator_map = {
                        'greater_than': '>',
                        'less_than': '<',
                        'greater_equal': '>=',
                        'less_equal': '<='
                    }
                    query_info.setdefault('where', {})
                    query_info['where']['conditions'] = [{
                        'type': 'comparison',
                        'column': column, 
                        'operator': operator_map[op_name], 
                        'value': value
                    }]
                    break
                    
            # Filtro por ano genérico
            year = self.extract_year(text)
            if year and 'where' not in query_info:
                date_columns = [col for col in self.schema[table] 
                               if any(date_term in col.lower() for date_term in 
                                     ['data', 'date', 'dt', 'ano', 'year'])]
                
                if date_columns:
                    query_info['where'] = {'conditions': [{'type': 'year', 'column': date_columns[0], 'value': year}]}
        
        # Detecta ORDER BY
        order_match = re.search(self.patterns['order_by'], text)
        if order_match:
            order_column = self.find_column_by_keyword(table, order_match.group(1))
            if order_column:
                direction = 'ASC'
                if order_match.group(2):
                    direction = 'DESC' if order_match.group(2).lower() in ['decrescente', 'desc'] else 'ASC'
                query_info['order_by'] = {'column': order_column, 'direction': direction}
                
        return query_info
        
    def generate_sql(self, query_info: Dict) -> str:
        """Gera SQL com suporte a JOINs e melhor formatação"""
        if 'error' in query_info:
            return query_info['error']
        
        table = query_info['table']
        query_type = query_info['type']
        
        # Constrói parte SELECT
        if query_type == 'select':
            select_clause = "SELECT *"
        elif query_type == 'count':
            select_clause = "SELECT COUNT(*)"
        elif query_type == 'average':
            column = query_info.get('column', 'nota')
            select_clause = f"SELECT AVG({column})"
        elif query_type == 'max':
            column = query_info.get('column', 'nota')
            select_clause = f"SELECT MAX({column})"
        elif query_type == 'min':
            column = query_info.get('column', 'nota')
            select_clause = f"SELECT MIN({column})"
        
        # Constrói parte FROM com JOINs
        from_clause = f"FROM {table}"
        if 'joins' in query_info:
            for join in query_info['joins']:
                from_clause += f" JOIN {join['table2']} ON {join['table1']}.{join['column1']} = {join['table2']}.{join['column2']}"
        
        sql = f"{select_clause} {from_clause}"
        
        # Adiciona condições WHERE com nova estrutura
        if 'where' in query_info and 'conditions' in query_info['where']:
            conditions = []
            
            for condition in query_info['where']['conditions']:
                if condition['type'] == 'equality':
                    column = condition['column']
                    value = condition['value']
                    if value.isdigit():
                        conditions.append(f"{column} = {value}")
                    else:
                        conditions.append(f"{column} LIKE '%{value}%'")
                        
                elif condition['type'] == 'comparison':
                    column = condition['column']
                    operator = condition['operator']
                    value = condition['value']
                    conditions.append(f"{column} {operator} {value}")
                        
                elif condition['type'] == 'year':
                    year_col = condition['column']
                    year_val = condition['value']
                    
                    if self.db.db_type == 'mysql':
                        conditions.append(f"YEAR({year_col}) = {year_val}")
                    else:
                        conditions.append(f"EXTRACT(YEAR FROM {year_col}) = {year_val}")
                        
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
                
        # Suporte à estrutura WHERE antiga (compatibilidade)
        elif 'where' in query_info:
            conditions = []
            
            if 'column' in query_info['where']:
                column = query_info['where']['column']
                value = query_info['where']['value']
                if value.isdigit():
                    conditions.append(f"{column} = {value}")
                else:
                    conditions.append(f"{column} LIKE '%{value}%'")
            
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
        
        # Adiciona ORDER BY
        if 'order_by' in query_info:
            order_info = query_info['order_by']
            sql += f" ORDER BY {order_info['column']} {order_info['direction']}"
                
        return sql


def display_results(results: List[Tuple], column_names: List[str]) -> None:
    """Exibe resultados com formatação melhorada"""
    if not results:
        print("❌ Nenhum resultado encontrado")
        return
    
    print(f"\n✅ Encontrados {len(results)} resultado(s)")
    print("=" * 50)
    
    if len(results) <= 20:
        print(tabulate(results, headers=column_names, tablefmt="grid"))
    else:
        print(f"Mostrando primeiros 20 de {len(results)} resultados:")
        print(tabulate(results[:20], headers=column_names, tablefmt="grid"))
        print(f"\n... e mais {len(results) - 20} resultado(s)")

def print_usage_guide():
    """Guia expandido com mais exemplos"""
    guide = """
🔍 === GUIA COMPLETO DO SISTEMA TEXT-TO-SQL ===

📋 TIPOS DE CONSULTA SUPORTADOS:

1️⃣ LISTAGEM BÁSICA
   💬 "mostre todos os alunos"
   💬 "listar professores"
   💬 "ver todas as matérias"
   🔧 SQL: SELECT * FROM [tabela]

2️⃣ CONTAGEM
   💬 "quantos alunos existem"
   💬 "contar professores de matemática"
   💬 "número total de notas"
   🔧 SQL: SELECT COUNT(*) FROM [tabela]

3️⃣ MÉDIAS
   💬 "média de notas dos alunos"
   💬 "qual a média das avaliações"
   🔧 SQL: SELECT AVG([coluna]) FROM [tabela]

4️⃣ VALORES MÁXIMOS/MÍNIMOS
   💬 "maior nota registrada"
   💬 "menor idade dos alunos"
   🔧 SQL: SELECT MAX/MIN([coluna]) FROM [tabela]

🔍 FILTROS AVANÇADOS:

📅 POR ANO: "notas de 2024", "alunos nascidos em 2000"
🎯 IGUALDADE: "alunos com nome João", "notas igual a 10"
📊 COMPARAÇÃO: "notas maiores que 8", "idades menores que 25"

💡 EXEMPLOS PRÁTICOS:
   "quantos alunos têm notas maiores que 7"
   "média de notas dos alunos em 2024"
   "professores com especialidade em matemática"
   "maior nota de cada aluno ordenado por nome"

⌨️ COMANDOS ESPECIAIS:
   'guia' - Mostra este guia
   'sair' - Encerra o programa
"""
    print(guide)

def show_tables_summary(schema: Dict[str, List[str]], column_types: Dict[str, Dict[str, str]]) -> None:
    """Exibe resumo melhorado das tabelas"""
    if not schema:
        print("\n❌ Nenhuma tabela disponível")
        return
        
    print("\n📊 === ESTRUTURA DO BANCO DE DADOS ===")
    
    for table in schema.keys():
        columns = schema[table]
        types = column_types.get(table, {})
        
        print(f"\n📋 {table.upper()} ({len(columns)} colunas)")
        print("-" * 40)
        
        for col in columns[:8]:  # Mostra até 8 colunas
            col_type = types.get(col, 'unknown')
            print(f"   • {col} ({col_type})")
        
        if len(columns) > 8:
            print(f"   ... e mais {len(columns) - 8} coluna(s)")

def main():
    print("🚀 === SISTEMA INTELIGENTE DE CONSULTAS SQL ===")
    print("Converte linguagem natural em SQL e executa no seu banco!")
    
    db = DatabaseConnection()
    
    # Interface melhorada para seleção de banco
    print("\n🔧 Selecione o tipo de banco de dados:")
    print("1️⃣  MySQL")
    print("2️⃣  PostgreSQL")
    
    choice = input("\n👉 Sua escolha [1]: ").strip() or "1"
    
    db_type = "mysql" if choice == "1" else "postgresql" if choice == "2" else "mysql"
    
    if choice not in ["1", "2"]:
        print("⚠️  Opção inválida. Usando MySQL como padrão.")
        db_type = "mysql"
    
    # Configuração automática
    host = "localhost"
    database = input(f"\n🗄️  Nome do banco de dados: ").strip()
    
    if db_type == "mysql":
        port = int(os.getenv("MYSQL_PORT", 3306))
        user = os.getenv("MYSQL_USER", "root")
        password = os.getenv("MYSQL_PASSWORD", "")
    else:
        port = int(os.getenv("POSTGRESQL_PORT", 5432))
        user = os.getenv("POSTGRESQL_USER", "postgres")
        password = os.getenv("POSTGRESQL_PASSWORD", "postgres")
        
    print(f"\n🔄 Conectando ao {db_type.upper()} em {host}:{port}...")
    
    if not db.connect(db_type, host, user, password, database, port):
        print(f"\n❌ Falha na conexão com '{database}'")
        print(f"\n🔧 Verificações necessárias:")
        print(f"   • Serviço {db_type} está rodando?")
        print(f"   • Usuário '{user}' existe?")
        print(f"   • Senha está correta?")
        print(f"   • Banco '{database}' existe?")
        return
    
    text_to_sql = TextToSQL(db)
    print("\n✅ Sistema pronto! Digite suas consultas em português.")
        
    while True:
        show_tables_summary(db.schema, db.column_types)
        
        print(f"\n💬 Digite sua consulta (ou 'sair' para sair, 'guia' para ajuda):")
        query = input("👉 ").strip()
        
        if query.lower() in ('sair', 's', 'exit', 'quit', 'q'):
            print("👋 Até logo!")
            break
        elif query.lower() in ('guia', 'guide', 'help', 'ajuda'):
            print_usage_guide()
            continue
        elif not query:
            continue
            
        print(f"\n🔄 Processando: '{query}'")
        
        query_info = text_to_sql.parse_query(query)
        sql = text_to_sql.generate_sql(query_info)
        
        print(f"\n🔧 SQL gerado:")
        print(f"   {sql}")
        
        if sql and not sql.startswith("Não"):
            results, column_names = db.execute_query(sql)
            display_results(results, column_names)
        else:
            print(f"❌ {sql}")
    
    db.disconnect()
    
if __name__ == "__main__":
    main()