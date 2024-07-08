import unittest, hexbugs as hb


class TestGenes(unittest.TestCase):

    def test_gene_init(self):
        gene = hb.Gene()
        self.assertEqual(gene.value, 0x000000)

    def test_gene_mutate(self):
        gene = hb.Gene()
        gene.mutate()
        self.assertNotEqual(gene.value, 0x000000)

    def test_gene_values(self):
        gene = hb.Gene(0x120000)
        self.assertEqual(gene.input_node, 0x1)
        self.assertEqual(gene.output_node, 0x2)
        self.assertEqual(gene.weight, -1.0)



if __name__ == '__main__':
    unittest.main()